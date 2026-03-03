
# Llama 8B Training with DeepSpeed and DFTracer Aggr on Tuolumne LLNL

## Create venv:
```bash
python3 -m venv your_venv_name
source your_venv_name/bin/activate
```

## Install torch etc:
```bash
pip install torch==2.9.1+rocm6.3 --index-url [https://download.pytorch.org/whl/rocm6.3](https://download.pytorch.org/whl/rocm6.3)
```

## Install DFTracer:
```bash
DFTRACER_VERSION=paper/dfprofiler
pip install git+https://github.com/LLNL/dftracer.git@${DFTRACER_VERSION}
```
## Environment Dependencies

---
Configured for ROCm 6.3:

    * `torch==2.9.1+rocm6.3`
    * `deepspeed==0.18.4`
    * `transformers==4.57.3`
    * `datasets==4.4.2`
    * `accelerate==1.12.0`
    * `trl==0.29.0`
    * `peft==0.18.1`
    * `numpy==2.3.5`
    * `pyarrow==23.0.1`
    * `safetensors==0.7.0`
    * `tokenizers==0.22.2`
    * `ninja==1.13.0`

---

## Dataset Preparation

Create a script that looks like this and run it before training:

```python
from datasets import load_dataset
import os

# Define your Lustre path
data_path = "/p/lustre5/${USER}/hf_datasets/teknium___open_hermes-2.5"

print("Downloading dataset to Lustre...")
# This downloads from HuggingFace and caches it locally
dataset = load_dataset("teknium/OpenHermes-2.5")

# Save to disk for fast local loading during training
dataset.save_to_disk(data_path)
