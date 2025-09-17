import os
import torch
import argparse
import deepspeed
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed.runtime.checkpoint_engine.decoupled_checkpoint_engine import DecoupledCheckpointEngine

import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # ------------------ Argument Parsing ------------------ #
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepspeed_config', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, default='bigscience/bloom-3b')
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./bloom3b-decoupled-finetuned')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()

    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)

    # ------------------ Register DecoupledCheckpointEngine ------------------ #
    deepspeed.runtime.checkpoint_engine.decoupled_checkpoint_engine.DecoupledCheckpointEngine = DecoupledCheckpointEngine

    # ------------------ Model and Tokenizer ------------------ #
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                device_map=None,
                                                low_cpu_mem_usage=True,
                                                torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # ------------------ Define Optimizer ------------------ #
    # # This is the new code you need to add to your script
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8)
    
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

    # ------------------ DeepSpeed Engine Initialization ------------------ #
    ds_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    print(f"DeepSpeed Config: {ds_engine.config}")

    global_rank = ds_engine.global_rank


    # ------------------ Checkpoint Engine Print Statement ------------------ #
    if ds_engine.checkpoint_engine:
        print(f"[Rank {global_rank}] Using checkpoint engine: {type(ds_engine.checkpoint_engine).__name__}")
    else:
        print(f"[Rank {global_rank}] No checkpoint engine detected. Using default behavior.")

    model.train()
    
    # ------------------ Example Checkpoint Save ------------------ #
    ckpt_dir = os.path.join(args.output_dir, "checkpoint-1")

    # # ------------------ Training Loop ------------------ #
    save_interval = 500
    latest_epoch = 0

    for epoch in range(latest_epoch, args.epochs):
        print(f"Entering epoch {epoch+1}")

        for i, block in enumerate(train_blocks):
            inputs = block.unsqueeze(0).to(ds_engine.device)
            labels = inputs.clone()
            outputs = ds_engine(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            ds_engine.step()

            if i % 50 == 0 and ds_engine.global_rank == 0:
                print(f"[Rank 0] Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

            checkpoint_now = (i % 50 == 0) or ((i + 1) % save_interval == 0) or (i + 1 == len(train_blocks))
            if checkpoint_now:
                # Start timer before the save call
                start_time = time.time()
                ds_engine.save_checkpoint(save_dir=ckpt_dir, tag="my_decoupled_ckpt")
                save_duration = time.time() - start_time
                print(f"Time to initiate save: {save_duration:.2f} seconds")

                ds_engine.checkpoint_engine.cleanup()

                total_duration = time.time() - start_time
                print(f"Total time to save and cleanup: {total_duration:.2f} seconds")

