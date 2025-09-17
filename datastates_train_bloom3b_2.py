import argparse
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.optim as optim
import time
from datastates.ckpt import CkptEngine
from datastates.llm import Checkpointing
import argparse
import json

# Parse arguments for DeepSpeed integration
parser = argparse.ArgumentParser()
parser.add_argument('--deepspeed_config', type=str, required=True, help='Path to DeepSpeed config file')
parser.add_argument('--model_name_or_path', type=str, default='bigscience/bloom-3b', help='HuggingFace model name or path')
parser.add_argument('--train_file', type=str, required=True, help='Path to a plain text training file')
parser.add_argument('--output_dir', type=str, default='./bloom3b-finetuned')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by deepspeed/torch.distributed')
parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Batch size per device/GPU')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--master_port', type=int, default=None, help='Master port for distributed training')
parser.add_argument('--resume_from', type=str, default=None, help="Resume training from this checkpoint tag (e.g., 'epoch_3'). If not set, start from scratch.")
args = parser.parse_args()

args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
if args.local_rank is not None and args.local_rank >= 0:
    torch.cuda.set_device(args.local_rank)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    device_map=None,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

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
print(f"Number of blocks: {len(train_blocks)}")

# Initialize DeepSpeed
ds_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=args.deepspeed_config
)

with open(args.deepspeed_config, "r") as f:
    ds_config = json.load(f)



# Initialize DataStates checkpointing engine
ckpt_engine = Checkpointing(runtime_config=ds_config.get("datastates_ckpt", {}), rank=ds_engine.global_rank)

save_interval = 500
model.train()

ckpt_dir = os.path.join(args.output_dir, "checkpoints")
if ds_engine.global_rank == 0:
    os.makedirs(ckpt_dir, exist_ok=True)


print(f"Starting training for {args.epochs} epochs", flush=True)


start_epoch = 0
start_step = 0

if args.resume_from is not None:
    tag_dir = os.path.join(ckpt_dir, args.resume_from)
    
    model_path = os.path.join(tag_dir, "mp_rank_00_model_states.pt")
    
    # Each rank loads its own sharded optimizer state file
    optim_path = os.path.join(tag_dir, f"zero_pp_rank_{ds_engine.global_rank}_mp_rank_00_optim_states.pt")

    # Load model state (rank 0 handles the model state)
    if ds_engine.global_rank == 0:
        if os.path.exists(model_path):
            print(f"[Rank 0] Loading model state from {model_path}", flush=True)
            model_state = ckpt_engine.load(model_path)
            ds_engine.load_state_dict(model_state["model"])
            print(f"[Rank 0] Finished loading model state.", flush=True)
        else:
            print(f"[Rank 0] No model checkpoint found at {model_path}.", flush=True)
    
    # Wait for all ranks to ensure rank 0 has loaded the model
    torch.distributed.barrier()

    # All ranks load their respective optimizer states
    if os.path.exists(optim_path):
        print(f"[Rank {ds_engine.global_rank}] Loading optimizer state from {optim_path}", flush=True)
        optim_state_dict = ckpt_engine.load(optim_path)
        # DeepSpeed's ZeroOptimizer needs to be loaded separately.
        # It handles the sharding and gathering internally.
        # optimizer.load_state_dict(optim_state_dict["optimizer"])
        print(f"[Rank {ds_engine.global_rank}] Finished loading optimizer state.", flush=True)
    else:
        print(f"[Rank {ds_engine.global_rank}] No optimizer state found at {optim_path}.", flush=True)
    
    # Wait for all ranks to finish loading their optimizer state
    torch.distributed.barrier()

    # Parse epoch/step from tag (This part of your code is correct)
    latest_tag = args.resume_from
    if latest_tag.startswith("epoch_"):
        start_epoch = int(latest_tag.split("_")[1]) - 1
        start_step = 0
    elif latest_tag.startswith("saveint_"):
        start_step = int(latest_tag.split("_")[1])
    elif latest_tag.startswith("step_"):
        start_step = int(latest_tag.split("_")[1])
    
    print(f"[Rank {ds_engine.global_rank}] Resuming from epoch {start_epoch+1}, step {start_step}", flush=True)

else:
    print(f"[Rank {ds_engine.global_rank}] No checkpoint found, starting from scratch.", flush=True)

for epoch in range(start_epoch, args.epochs):
    print(f"Entering epoch {epoch+1}", flush=True)
    total_tokens = 0
    for i, block in enumerate(train_blocks):
        if epoch == start_epoch and i < start_step:
            continue  # Skip already trained steps
        inputs = block.unsqueeze(0).to(ds_engine.device)
        labels = inputs.clone()
        outputs = ds_engine(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        ds_engine.step()
        total_tokens += block.numel()

        if i % 50 == 0 and ds_engine.global_rank == 0:
            print(f"[Rank 0] Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}", flush=True)

        # Decide if checkpoint should be saved
        checkpoint_now = (i % 50 == 0) or ((i + 1) % save_interval == 0) or (i + 1 == len(train_blocks))
        if checkpoint_now:
            # Determine tag/folder name
            if (i + 1) == len(train_blocks):
                tag = f"epoch_{epoch+1}"
            elif (i + 1) % save_interval == 0:
                tag = f"saveint_{i+1}"
            else:
                tag = f"step_{i+1}"

            # Create folder for this checkpoint
            tag_dir = os.path.join(ckpt_dir, tag)
            if ds_engine.global_rank == 0:
                os.makedirs(tag_dir, exist_ok=True)

            torch.distributed.barrier()  # make sure all ranks wait
            # Save model (rank 0 only)
            epoch_start_time = time.time()
            if ds_engine.global_rank == 0:
                model_path = os.path.join(tag_dir, "mp_rank_00_model_states.pt")
                ckpt_engine.save(state_dict={"model": ds_engine.state_dict()}, path=model_path)
                print(f"[Rank 0] Saved model state to {model_path}", flush=True)

            # Save optimizer shard (all ranks)
            optim_path = os.path.join(tag_dir, f"zero_pp_rank_{ds_engine.global_rank}_mp_rank_00_optim_states.pt")
            ckpt_engine.save(state_dict={"optimizer": optimizer.state_dict()}, path=optim_path)
            print(f"[Rank {ds_engine.global_rank}] Saved optimizer state to {optim_path}", flush=True)

            # Ensure all async saves are done
            ckpt_engine.wait()

            if ds_engine.global_rank == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"[Rank 0] Checkpoint {tag} saved. Save time: {epoch_time:.2f} seconds", flush=True)
            