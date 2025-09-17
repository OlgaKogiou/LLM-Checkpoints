import os
import time
import torch
import argparse
import deepspeed
from types import SimpleNamespace
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace
import deepspeed
from deepspeed.ops.op_builder import AsyncIOBuilder, GDSBuilder
from deepspeed.io import MockFileWriter, PyFileWriter, FastFileWriter, FastFileWriterConfig
from deepspeed.accelerator import get_accelerator

# Add the JIT compilation code here, right after imports
# This prevents deadlocks by ensuring extensions are pre-loaded
try:
    from deepspeed.ops.op_builder import AsyncIOBuilder
    AsyncIOBuilder().load(verbose=False)
except ImportError:
    pass

try:
    from deepspeed.ops.op_builder import UtilsBuilder
    UtilsBuilder().load(verbose=False)
except ImportError:
    pass

# ------------------ Argument Parsing ------------------ #
parser = argparse.ArgumentParser()
parser.add_argument('--deepspeed_config', type=str, required=True)
parser.add_argument('--model_name_or_path', type=str, default='bigscience/bloom-3b')
parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='./bloom3b-finetuned')
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--per_device_train_batch_size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--resume_from', type=str, default=None)
args = parser.parse_args()

args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
if args.local_rank is not None and args.local_rank >= 0:
    torch.cuda.set_device(args.local_rank)

# ------------------ Model and Tokenizer ------------------ #
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    device_map=None,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

# ------------------ Dataset ------------------ #
def load_dataset_stream(file_path, tokenizer, block_size=512, max_blocks=None):
    blocks = []
    buffer = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip():
                continue
            tokens = tokenizer(line, return_tensors='pt', truncation=False)["input_ids"].squeeze(0)
            buffer.append(tokens)
            while sum(b.numel() for b in buffer) >= block_size:
                concat = torch.cat(buffer)
                blocks.append(concat[:block_size])
                buffer = [concat[block_size:]] if concat.numel() > block_size else []
                if max_blocks is not None and len(blocks) >= max_blocks:
                    return blocks
    if buffer and (max_blocks is None or len(blocks) < max_blocks):
        concat = torch.cat(buffer)
        if concat.numel() >= 1:
            concat = torch.nn.functional.pad(concat, (0, block_size - concat.numel()), value=tokenizer.pad_token_id)
            blocks.append(concat)
    return blocks

train_blocks = load_dataset_stream(args.train_file, tokenizer, block_size=512, max_blocks=450)

# ------------------ DeepSpeed Engine ------------------ #
# This initializes the training components, but we'll bypass its checkpointing.
ds_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=args.deepspeed_config
)

world_size = ds_engine.world_size
global_rank = ds_engine.global_rank

# ------------------ Prepare I/O components for FastFileWriter ------------------ #
with open(args.deepspeed_config, "r") as f:
    ds_config_dict = json.load(f)

# Use the aio_config from the DeepSpeed JSON
aio_config = ds_config_dict.get("aio_config", {})
aio_builder = AsyncIOBuilder().load()

h = aio_builder.aio_handle(
    block_size=aio_config.get("block_size", 1024*1024),
    queue_depth=aio_config.get("queue_depth", 8),
    single_submit=aio_config.get("single_submit", False),
    overlap_events=aio_config.get("overlap_events", False),
    intra_op_parallelism=aio_config.get("thread_count", 1)
)

io_buffer_size = ds_config_dict["checkpoint_config"]["writer"]["io_buffer_size"]
pinned_memory = torch.empty(io_buffer_size, dtype=torch.uint8, device='cpu').pin_memory()

# ------------------ Prepare checkpoint directory ------------------ #
ckpt_dir = os.path.join(args.output_dir, "checkpoints")
if ds_engine.global_rank == 0:
    os.makedirs(ckpt_dir, exist_ok=True)

# ------------------ Resume from checkpoint ------------------ #
latest_epoch = 0
if args.resume_from is not None:
    print(f"[Rank {global_rank}] Resuming from checkpoint: {args.resume_from}")
    load_start = time.time()
    checkpoint_path = os.path.join(ckpt_dir, f"{args.resume_from}.pt")
    if os.path.exists(checkpoint_path):
        # We're manually loading the checkpoint with a standard torch.load()
        state_dict = torch.load(checkpoint_path, map_location=ds_engine.device)
        model.load_state_dict(state_dict)
        latest_epoch = int(args.resume_from.split("_")[1])
        load_time = time.time() - load_start
        print(f"[Rank {global_rank}] Resumed checkpoint {args.resume_from} in {load_time:.2f}s")
    else:
        load_time = time.time() - load_start
        print(f"[Rank {global_rank}] No checkpoint found, starting from scratch (took {load_time:.2f}s)")

model.train()

# ------------------ Training Loop ------------------ #
save_interval = 500

for epoch in range(latest_epoch, args.epochs):
    print(f"Entering epoch {epoch+1}")
    for i, block in enumerate(train_blocks):
        inputs = block.unsqueeze(0).to(ds_engine.device)
        labels = inputs.clone()
        outputs = ds_engine(inputs, labels=labels)
        loss = outputs.loss
        ds_engine.backward(loss)
        ds_engine.step()

        if i % 50 == 0:
            print(f"[Rank {global_rank}] Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")
            print(f"I/O Buffer Size: {io_buffer_size}", flush=True)
            num_writers = ds_config_dict["checkpoint_config"]["writer"].get("num_writers", 1)
            print(f"Number of Writers: {num_writers}", flush=True)

        checkpoint_now = (i > 0 and i % 50 == 0) or ((i + 1) % save_interval == 0) or (i + 1 == len(train_blocks))
        if checkpoint_now:
            save_tag = f"epoch_{epoch+1}_step_{i}"
            save_path = os.path.join(ckpt_dir, f"{save_tag}.pt")
            
            print(f"[Rank {global_rank}] Saving checkpoint {save_tag} with FastFileWriter")
            save_start = time.time()
            
            # Manually instantiate and use FastFileWriter
            fast_writer_config = SimpleNamespace(
                dnvme_handle=h,
                pinned_tensor=pinned_memory,
                double_buffer=ds_config_dict["checkpoint_config"]["writer"].get("io_buffer_double", True),
                num_parallel_writers=ds_config_dict["checkpoint_config"]["writer"].get("num_writers", 1),
                writer_rank=global_rank,
                global_rank=global_rank
            )
            
            ds_fast_writer = FastFileWriter(file_path=save_path, config=fast_writer_config)
            
            # Save the model and optimizer states in a single dictionary
            checkpoint_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            # checkpoint_state = {'model': optimizer.state_dict()}
            torch.save(f=ds_fast_writer, obj=checkpoint_state)
            ds_fast_writer.close()
            
            save_time = time.time() - save_start
            print(f"[Rank {global_rank}] Saved checkpoint {save_tag} in {save_time:.2f}s")
