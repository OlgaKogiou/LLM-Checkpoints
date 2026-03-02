#!/bin/bash

# 1. Environment Setup
source /usr/workspace/kogiou1/venvs/deepspeed_venv_dftracer/bin/activate
module load rocm/6.3.0

# 1. Clear the failed build again
rm -rf /g/g92/kogiou1/.cache/torch_extensions/

# 2. Set explicit paths for BOTH the compiler and the C++ wrapper
export CC=$(which gcc)
export CXX=$(which g++)
export PATH=$(dirname $(which gcc)):$PATH
export PATH=$(python -c "import sys; import os; print(os.path.dirname(sys.executable))"):$PATH
export PATH=/usr/workspace/kogiou1/venvs/deepspeed_venv_dftracer/bin:$PATH
export CMAKE_BIN_DIR=/usr/workspace/kogiou1/venvs/deepspeed_venv_dftracer/bin

echo "This is path:"
echo $PATH
# Verify ninja is found
echo "This is ninja"
which ninja
echo "This is gcc:"
which gcc
gcc --version

# 2. THE CRITICAL FIXES FOR AMD
export DS_ACCELERATOR=cuda      
export HIP_VISIBLE_DEVICES=0,1,2,3
export DS_SKIP_CUDA_CHECK=1     
export HIP_PLATFORM=amd         
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export ROCM_PATH=/opt/rocm-6.3.0
export CUDA_HOME=$ROCM_PATH

# 3. Path Overrides (Crucial for Lustre usage)
export HF_HOME="/p/lustre5/kogiou1/hf_cache"
export HF_DATASETS_CACHE="/p/lustre5/kogiou1/hf_datasets"
export HF_DATASETS_TRUST_REMOTE_CODE=True

export DFTRACER_INSTALLED=/usr/workspace/kogiou1/venvs/deepspeed_venv_dftracer/lib/python3.11/site-packages/dftracer
export LD_LIBRARY_PATH=$DFTRACER_INSTALLED/lib:$DFTRACER_INSTALLED/lib64:$LD_LIBRARY_PATH
export DFTRACER_LOG_FILE=/usr/workspace/iopp/kogiou1/dftracer_aggr_events/llama-3-8b/32_nodes/default/llama-3-8b
export DFTRACER_DATA_DIR=/p/lustre5/kogiou1/hf_datasets/teknium___open_hermes-2.5:/p/lustre5/kogiou1/llama-3-8b
export DFTRACER_ENABLE=1
export DFTRACER_INC_METADATA=1
export DFTRACER_TRACE_COMPRESSION=1
export DFTRACER_INIT=FUNCTION
export DFTRACER_BIND_SIGNALS=0
export DFTRACER_LOG_LEVEL=INFO
export DFTRACER_WRITE_BUFFER_SIZE=4096 # writes traces in 4096 bytes chunks
export DFTRACER_DISABLE_STDIO=0

# 4. Generate Hostfile for 2 nodes, 4 slots each
flux run -N 16 -n 16 hostname | sort -u | awk '{print $1 " slots=4"}' > dp_llama_hostfile

echo "--- Generated Hostfile ---"
cat dp_llama_hostfile
echo "--------------------------"

# 5. Network Setup
export MASTER_ADDR=$(head -n 1 dp_llama_hostfile | awk '{print $1}')
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

# 6. Launch Training
# Using your requested parameters: 2 nodes, 4 GPUs per node
deepspeed \
    --launcher pdsh \
    --hostfile dp_llama_hostfile \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --num_nodes 16 \
    --num_gpus 4 \
    train_llama3.py --deepspeed_config /usr/workspace/kogiou1/LLM_work/DeepSpeed/run/llama8b/training/ds_config_zero3.json