from torch.utils.data import Dataset
from datasets import load_dataset
import os
from pathlib import Path

import argparse
import deepspeed
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
import torch
import torch.distributed as dist
import time
import math
from dftracer.python import dftracer, ai


class OpenHermesDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, max_length=128, cache_dir=None):
        self.raw_data = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_data)

    @ai.data.item
    def __getitem__(self, idx):
        example = self.raw_data[idx]

        with ai.data.preprocess:
            text = str(example)
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )

        total_bytes = len(text.encode("utf-8"))
        ai.update(image_size=total_bytes, image_idx=idx)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    @ai.data.preprocess.derive("collate_fn")
    def __call__(self, features, return_tensors: str | None = None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "pt":
            return self.torch_call(features)
        if return_tensors == "np":
            return self.numpy_call(features)
        raise ValueError(f"Framework '{return_tensors}' not recognized!")


def format_duration(seconds: float) -> str:
    total_seconds = 0 if seconds <= 0 else max(1, math.ceil(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    if minutes == 0:
        return f"{remaining_seconds}s"
    return f"{minutes}m {remaining_seconds}s"


@ai
def run(args, data_folder_dir, output_dir):
    model_id = args.model_id
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
    )

    local_model = os.path.isdir(model_id)
    looks_like_path = model_id.startswith("/") or model_id.startswith(".")

    if looks_like_path and not local_model:
        raise FileNotFoundError(
            f"Model path does not exist: {model_id}. "
            "Use --model_id with an existing local directory or a valid Hugging Face repo id."
        )

    if args.local_rank <= 0:
        print(f"Using model source: {model_id}")
        print(
            f"Model source type: {'local directory' if local_model else 'huggingface hub repo'}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        local_files_only=local_model,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    custom_dataset = OpenHermesDataset(
        args.dataset_name,
        tokenizer,
        max_length=128,
        cache_dir=data_folder_dir,
    )
    collate_fn = CustomDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=args.low_cpu_mem_usage,
        use_cache=False,
        token=hf_token,
        local_files_only=local_model,
    )

    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=custom_dataset,
        collate_fn=collate_fn,
    )

    starting_epoch = 0
    global_step = 0
    batches_to_skip = 0

    if args.resume_from_checkpoint:
        if args.local_rank <= 0:
            print(
                f"Attempting to resume from checkpoint: {args.resume_from_checkpoint}"
            )

        _, client_state = model_engine.load_checkpoint(args.resume_from_checkpoint)

        if client_state is not None:
            starting_epoch = client_state.get("epoch", 0)
            global_step = client_state.get("global_step", 0)
            batches_to_skip = client_state.get("batch_index", 0) + 1

            if args.local_rank <= 0:
                print(
                    f"Successfully loaded checkpoint state! Resuming from Epoch {starting_epoch}, "
                    f"Global Step {global_step}, skipping {batches_to_skip} batches."
                )

    num_epochs = args.num_epochs
    save_steps = args.save_steps
    should_stop = args.max_global_step > 0 and global_step >= args.max_global_step
    step_count = 0

    @ai.dataloader.fetch
    def fetch_next_batch(iterator):
        return next(iterator)

    for epoch in ai.pipeline.epoch.iter(
        range(starting_epoch, num_epochs), include_iter=False
    ):
        if should_stop:
            break

        epoch_start_time = time.perf_counter()
        data_iterator = iter(training_dataloader)
        batch_idx = 0

        while True:
            step_start_time = time.perf_counter()
            ai.update(step=batch_idx, epoch=epoch, args={"global_step": global_step})

            if should_stop:
                break

            try:
                batch = fetch_next_batch(data_iterator)
            except StopIteration:
                break

            if epoch == starting_epoch and batch_idx < batches_to_skip:
                batch_idx += 1
                continue

            ai.compute.start()
            with ai.device.transfer:
                batch = {
                    key: value.to(model_engine.local_rank)
                    for key, value in batch.items()
                }

            with ai.compute.forward:
                outputs = model_engine(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss

            with ai.compute.backward:
                model_engine.backward(loss)
                model_engine.step()

            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1
                step_elapsed = time.perf_counter() - step_start_time

                if args.local_rank <= 0:
                    print(
                        f"Epoch: {epoch} | Global Step: {global_step} | Loss: {loss.item():.4f}"
                    )
                    print(f"Step time: {format_duration(step_elapsed)}")

                if args.max_global_step > 0 and global_step >= args.max_global_step:
                    should_stop = True

                if global_step > 0 and global_step % save_steps == 0:
                    tag = f"global_step_{global_step}"

                    client_state = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "batch_index": batch_idx,
                    }
                    with ai.checkpoint.capture:
                        if args.local_rank <= 0:
                            print(
                                f"Saving checkpoint at global step {global_step} with tag '{tag}'..."
                            )
                        model_engine.save_checkpoint(
                            save_dir=output_dir,
                            tag=tag,
                            client_state=client_state,
                        )

                    # with ai.checkpoint.restart:
                    #     if args.local_rank <= 0:
                    #         print(f"Restarting checkpoint at global step {global_step}...")
                    #     model_engine.load_checkpoint(load_dir=output_dir, tag=tag)

                    # if args.local_rank <= 0:
                    # print("Read complete! Resuming training...")

            ai.compute.stop()

            batch_idx += 1
            step_count += 1

            if args.max_step > 0 and step_count >= args.max_step:
                print(f"Reached max raw step count of {args.max_step}. Stopping training loop.")
                should_stop = True
            if args.max_global_step > 0 and global_step >= args.max_global_step:
                print(f"Reached max global step of {args.max_global_step}. Stopping training loop.")
                should_stop = True

        if args.local_rank <= 0:
            epoch_elapsed = time.perf_counter() - epoch_start_time
            print(f"Epoch time: {format_duration(epoch_elapsed)}")

    if args.local_rank <= 0:
        print("Training complete! Saving final model...")
    with ai.checkpoint.capture:
        model_engine.save_checkpoint(save_dir=output_dir, tag="final_model")


def get_args():
    default_local_model = str((Path(__file__).resolve().parent / "llama-3-8b"))

    parser = argparse.ArgumentParser(
        description="Pure DeepSpeed Llama 3 Training with Resume"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=default_local_model,
        help="Model path or Hugging Face repo id",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint folder",
    )
    parser.add_argument(
        "--max-global-step",
        "--max_global_step",
        dest="max_global_step",
        type=int,
        default=-1,
        help="[DEBUG ONLY] Stop when global optimizer step reaches this value",
    )
    parser.add_argument(
        "--max-step",
        "--max_step",
        dest="max_step",
        type=int,
        default=-1,
        help="[DEBUG ONLY] Stop when raw training loop step reaches this value",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Total number of training epochs"
    )
    parser.add_argument(
        "--save_steps", type=int, default=5, help="Save checkpoint every N global steps"
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Enable low CPU memory model loading in Hugging Face from_pretrained",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/p/lustre5/iopp/rayandrew/dfprofiler/results/llama-3-8b",
        help="Root directory where run-specific outputs are written",
    )
    parser.add_argument(
        "--data_folder_dir",
        type=str,
        default="/p/lustre5/sinurat1/dataset/ml-workloads/hf_datasets",
        help="Directory used for Hugging Face dataset cache files",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="teknium/OpenHermes-2.5",
        help="Hugging Face dataset identifier to train on",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.num_epochs < 1:
        raise ValueError(f"--num_epochs must be >= 1, got {args.num_epochs}")
    if args.save_steps < 1:
        raise ValueError(f"--save_steps must be >= 1, got {args.save_steps}")
    if args.max_global_step < -1:
        raise ValueError(f"--max-global-step must be >= -1, got {args.max_global_step}")
    if args.max_step < -1:
        raise ValueError(f"--max-step must be >= -1, got {args.max_step}")

    base_output_dir = args.output_root
    data_folder_dir = args.data_folder_dir
    trace_logs_dir = f"{base_output_dir}/logs"
    trace_output_dir = (
        f"{trace_logs_dir}/app-{args.local_rank}-of-{torch.cuda.device_count()}"
    )
    output_dir = f"{base_output_dir}/checkpoints"

    os.makedirs(data_folder_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    df_logger = dftracer.initialize_log(
        trace_output_dir, f"{data_folder_dir}:{output_dir}", -1
    )

    try:
        run(args, data_folder_dir, output_dir)
    finally:
        df_logger.finalize()

        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
