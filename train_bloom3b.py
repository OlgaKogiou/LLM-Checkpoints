import os
import time
import torch
import argparse
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Prefer LOCAL_RANK from environment (set by srun/DeepSpeed) when available.
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

ds_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=args.deepspeed_config
)

model.train()

save_interval = 500
ckpt_dir = os.path.join(args.output_dir, "checkpoints")
if ds_engine.global_rank == 0:
    os.makedirs(ckpt_dir, exist_ok=True)

# --- Resume logic ---
latest_epoch = 0
if args.resume_from is not None:
    print(f"[Rank {ds_engine.global_rank}] Attempting to resume from checkpoint: {args.resume_from}", flush=True)
    load_start = time.time()
    load_success, client_state = ds_engine.load_checkpoint(ckpt_dir, tag=args.resume_from)
    load_time = time.time() - load_start
    if ds_engine.global_rank == 0:
        print(f"[Rank 0] Checkpoint loaded in {load_time:.2f}s", flush=True)
    if load_success:
        latest_epoch = int(args.resume_from.split("_")[1])
        print(f"[Rank {ds_engine.global_rank}] Successfully resumed from {args.resume_from}, continuing at epoch {latest_epoch+1}")
    else:
        print(f"[Rank {ds_engine.global_rank}] Failed to load checkpoint {args.resume_from}, starting from scratch")

for epoch in range(latest_epoch, args.epochs):
    print(f"Entering epoch {epoch+1}", flush=True)

    for i, block in enumerate(train_blocks):
        # Move input block to the correct device and add batch dimension
        inputs = block.unsqueeze(0).to(ds_engine.device)
        labels = inputs.clone()

        outputs = ds_engine(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        ds_engine.step()

        if i % 50 == 0 and ds_engine.global_rank == 0:
            print(f"[Rank 0] Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}", flush=True)

        checkpoint_now = (i % 50 == 0) or ((i + 1) % save_interval == 0) or (i + 1 == len(train_blocks))
        if checkpoint_now:
            print(f"[Rank {ds_engine.global_rank}] Saving checkpoint at epoch {epoch+1}, step {i}", flush=True)
            save_start = time.time()
            ds_engine.save_checkpoint(ckpt_dir, tag=f"epoch_{epoch+1}")
            save_time = time.time() - save_start
            if ds_engine.global_rank == 0:
                print(f"[Rank 0] Checkpoint saved in {save_time:.2f}s", flush=True)