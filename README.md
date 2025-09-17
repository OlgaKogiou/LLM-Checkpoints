# DeepSpeed Bloom 3B Training on sxm1

This guide provides instructions for running the DeepSpeed training scripts on the `sxm1` server using 4 NVIDIA V100 GPUs.

## Environment Setup
Use **Python 3.10** for this project
- dependencies1.txt --> for the conda env
- Olga's env: new_deepspeed

### CUDA Environment Variables
Ensure you have **CUDA Toolkit 11.8** available on your system. 
```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### DeepSpeed Torch Checkpoint Engine:
Save Checkpoints every 50 and 500 steps: 
Time: 46s
```bash
deepspeed --num_gpus 4 run_any_engine.py --deepspeed_config ds_torch_config.json --train_file input_data.txt --output_dir ./bloom3b-torch/
```

### DeepSpeed Fast Checkpoint Engine:
Save Checkpoints every 50 and 500 steps: 
Time: 460s
```bash
deepspeed --num_gpus 4 run_any_engine.py --deepspeed_config ds_torch_config.json --train_file input_data.txt --output_dir ./bloom3b-fast/
```

### DeepSpeed Decoupled Checkpoint Engine:
Save Checkpoints every 50 and 500 steps: 
Time: 40s (Async)
```bash
deepspeed --num_gpus 4 run_any_engine.py --deepspeed_config ds_decoupled_config.json --train_file input_data.txt --output_dir ./bloom3b-decoupled/
```

### Raw Test of Fast Writer (1 per checkpoint)
Save Checkpoints every 50 and 500 steps: 
Time: 460s
```bash
deepspeed --num_gpus 4 test_fast_writter_bloom_3b.py --deepspeed_config ds_config_zero2.json --train_file input_data.txt --output_dir ./bloom3b-test
```
