from torch.utils.data import Dataset
from datasets import load_dataset
import os

os.environ["HF_HOME"] = "/p/lustre5/kogiou1/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/p/lustre5/kogiou1/hf_datasets/"

os.environ["DFTRACER_INSTALLED"] = "/usr/workspace/kogiou1/venvs/deepspeed_venv_dftracer/lib/python3.11/site-packages/dftracer"
os.environ["DFTRACER_LOG_FILE"] = "/usr/workspace/iopp/kogiou1/dftracer_aggr_events/llama-3-8b/32_nodes/default/llama-3-8b"
os.environ["DFTRACER_DATA_DIR"] = "/p/lustre5/kogiou1/hf_datasets/teknium___open_hermes-2.5:/p/lustre5/kogiou1/llama-3-8b"
os.environ["DFTRACER_ENABLE"] = "1"
os.environ["DFTRACER_INC_METADATA"] = "1"
os.environ["DFTRACER_TRACE_COMPRESSION"] = "1"
os.environ["DFTRACER_INIT"] = "FUNCTION"   # Using function-level tracing instead of LD_PRELOAD
os.environ["DFTRACER_BIND_SIGNALS"] = "0"
os.environ["DFTRACER_LOG_LEVEL"] = "INFO"
os.environ["DFTRACER_WRITE_BUFFER_SIZE"] = "4096" 
os.environ["DFTRACER_DISABLE_STDIO"] = "0"
os.environ["LD_LIBRARY_PATH"] = "/usr/workspace/kogiou1/venvs/deepspeed_venv_dftracer/lib/python3.11/site-packages/dftracer/lib:/usr/workspace/kogiou1/venvs/deepspeed_venv_dftracer/lib/python3.11/site-packages/dftracer/lib64"

import shutil
import argparse
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import torch
from dftracer.python import dftracer, ai

class OpenHermesDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, max_length=128):
        """
        Initializes the dataset fetching the pre-cached OpenHermes-2.5 data.
        """
        # Because it's fully cached, this will instantly load from disk
        self.raw_data = load_dataset(dataset_name, split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_data)

    # DFTRACER 5: data item
    @ai.data.item
    def __getitem__(self, idx):
        """
        Fetches a single row from the .arrow file on the fly.
        """
        example = self.raw_data[idx]
        
        # Force the read into memory by casting the row to a string
        text = str(example)
        
        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length, 
            padding=False, 
            return_tensors=None 
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }


def main():
    
    # --- TRACE: Set up dftracer ---
    trace_output_dir = "/usr/workspace/iopp/kogiou1/dftracer_aggr_events/llama-3-8b/32_nodes/default"
    data_folder_dir = "/p/lustre5/kogiou1/hf_datasets/teknium___open_hermes-2.5"
    
    os.makedirs(trace_output_dir, exist_ok=True)
    model_id = "meta-llama/Meta-Llama-3-8B"
    output_dir = "/p/lustre5/kogiou1/llama-3-8b"
    os.makedirs(output_dir, exist_ok=True)
    
    # DFTRACER 1: AI 
    # df_logger = dftracer.initialize_log(trace_output_dir, data_folder_dir, -1)
    rank = int(os.environ.get("RANK", 0))
    df_logger = dftracer.initialize_log(trace_output_dir, f"{data_folder_dir}:{output_dir}", -1)

    venv_bin_path = "/usr/workspace/kogiou1/venvs/deepspeed_venv_dftracer/bin"
    if venv_bin_path not in os.environ["PATH"]:
        os.environ["PATH"] = venv_bin_path + os.pathsep + os.environ["PATH"]

    ninja_check = shutil.which("ninja")
    print(f"--- Node {os.uname()[1]} confirms ninja is at: {ninja_check} ---")
    
    os.environ["CC"] = "/usr/tce/bin/gcc"
    os.environ["CXX"] = "/usr/tce/bin/g++"
    print(f"--- Node {os.uname()[1]} using CC: {os.environ.get('CC')} ---")

    parser = argparse.ArgumentParser(description="Pure DeepSpeed Llama 3 Training with Resume")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="Path to the DeepSpeed checkpoint folder to resume from")
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # --- 1. Load Tokenizer & Initialize Custom Dataset ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    custom_dataset = OpenHermesDataset("teknium/OpenHermes-2.5", tokenizer, max_length=128)
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # --- 2. Load Model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        use_cache=False
    )

    # --- 3. Initialize DeepSpeed Engine ---
    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=custom_dataset,
        collate_fn=collate_fn
    )

    # --- 4. The Restart Interface ---
    starting_epoch = 0
    global_step = 0
    batches_to_skip = 0

    if args.resume_from_checkpoint:
        if args.local_rank <= 0:
            print(f"Attempting to resume from checkpoint: {args.resume_from_checkpoint}")
        
        _, client_state = model_engine.load_checkpoint(args.resume_from_checkpoint)
        
        if client_state is not None:
            starting_epoch = client_state.get('epoch', 0)
            global_step = client_state.get('global_step', 0)
            batches_to_skip = client_state.get('batch_index', 0) + 1
            
            if args.local_rank <= 0:
                print(f"Successfully loaded checkpoint state! Resuming from Epoch {starting_epoch}, "
                      f"Global Step {global_step}, skipping {batches_to_skip} batches.")

    # --- 5. Pure DeepSpeed Training Loop ---
    num_epochs = 1
    save_steps = 5

    # DFTRACER 6: Helper function to trace dataloader batch fetching
    @ai.dataloader.fetch
    def fetch_next_batch(iterator):
        return next(iterator)

    # DFTRACER 2: pipeline train
    # for epoch in range(starting_epoch, num_epochs):
    for epoch in ai.pipeline.epoch.iter(range(starting_epoch, num_epochs)):
        
        # Create a manual iterator so we can call fetch_next_batch() on it
        data_iterator = iter(training_dataloader)
        batch_idx = 0
        
        while True:
            try:
                # Trigger the fetch trace decorator
                batch = fetch_next_batch(data_iterator)
            except StopIteration:
                # Iterator is empty, epoch is finished
                break
            
            # Fast-forward dataloader if we are resuming mid-epoch
            if epoch == starting_epoch and batch_idx < batches_to_skip:
                batch_idx += 1
                continue
                
            # DFTRACER 7: compute block
            ai.compute.start()
            with ai.device.transfer:
                batch = {k: v.to(model_engine.local_rank) for k, v in batch.items()}

            # DFTRACER 8: compute forward
            with ai.compute.forward:
                outputs = model_engine(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss

            # DFTRACER 9: compute backward
            with ai.compute.backward:
                model_engine.backward(loss)
                model_engine.step()

            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1

                if args.local_rank <= 0:
                    print(f"Epoch: {epoch} | Global Step: {global_step} | Loss: {loss.item():.4f}")

                if global_step > 0 and global_step % save_steps == 0:
                    tag = f"global_step_{global_step}"
                    
                    client_state = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'batch_index': batch_idx
                    }
                    # DFTRACER 10: checkpoint capture
                    with ai.checkpoint.capture:
                        model_engine.save_checkpoint(
                            save_dir=output_dir, 
                            tag=tag,
                            client_state=client_state
                        )
                    
                    # DFTRACER 11: restart capture
                    with ai.checkpoint.restart:
                        model_engine.load_checkpoint(
                            load_dir=output_dir,
                            tag=tag
                        )
                    
                    if args.local_rank <= 0:
                        print("Read complete! Resuming training...")

            # DFTRACER 7: compute block
            ai.compute.stop()
            
            # Manually increment batch counter at the end of the loop
            batch_idx += 1

    # --- 6. Final Save ---
    if args.local_rank <= 0:
        print("Training complete! Saving final model...")
    model_engine.save_checkpoint(save_dir=output_dir, tag="final_model")

    # DFTRACER 1: AI
    df_logger.finalize()

if __name__ == "__main__":
    main()
