from datasets import load_dataset
import os

# Define your Lustre path
data_path = "/p/lustre5/sinurat1/dataset/ml-workloads/hf_datasets/teknium___open_hermes-2.5"

print("Downloading dataset to Lustre...")
# This downloads from HuggingFace and caches it locally
dataset = load_dataset("teknium/OpenHermes-2.5")

# Save to disk for fast local loading during training
dataset.save_to_disk(data_path)