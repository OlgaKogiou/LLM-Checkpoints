# DeepSpeed Bloom 3B Training on sxm1

This guide provides instructions for running the DeepSpeed training scripts on the `sxm1` server using 4 NVIDIA V100 GPUs.

## Environment Setup
Use **Python 3.10** for this project
- dependencies.txt --> for the conda env
- Olga's env: datastates-llm

### CUDA Environment Variables
Ensure you have **CUDA Toolkit 11.8** available on your system. 
```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Datastates LLM Run:
Save Checkpoints every 50 and 500 steps: 
Time: 1.93s
```bash
deepspeed --num_gpus 4 datastates_train_bloom3b.py --deepspeed_config ds_config_zero2_datastates.json --train_file input_data.txt --output_dir ./bloom3b-finetuned-datastates/
```

Load Checkpoints and continue Training:
Time: 33.38s
```bash
deepspeed --num_gpus 4 datastates_train_bloom3b.py --deepspeed_config ds_config_zero2_datastates.json --train_file input_data.txt --output_dir ./bloom3b-finetuned-datastates/ --resume_from epoch_1
```

### DeepSpeed Run:
Save Checkpoints every 50 and 500 steps: 
Time: 98.88s
```bash
deepspeed --num_gpus 4 train_bloom3b.py --deepspeed_config ds_config_zero2.json --train_file input_data.txt --output_dir ./bloom3b-finetuned/
```

Load Checkpoints and continue Training:
Time: 15.05s
```bash
deepspeed --num_gpus 4 train_bloom3b.py --deepspeed_config ds_config_zero2.json --train_file input_data.txt --output_dir ./bloom3b-finetuned/ --resume_from epoch_1
```

