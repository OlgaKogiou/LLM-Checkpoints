#!/bin/bash

# 1. Environment Setup
# source /usr/workspace/kogiou1/venvs/deepspeed_venv_dftracer/bin/activate
source /p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/bin/activate
module load rocm/6.3.0

# 1. Clear the failed build again
# rm -rf /g/g92/kogiou1/.cache/torch_extensions/
rm -rf /p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/.cache/torch_extensions/

# 2. Set explicit paths for BOTH the compiler and the C++ wrapper
export CC=$(which gcc)
export CXX=$(which g++)
export PATH=$(dirname $(which gcc)):$PATH
export PATH=$(python -c "import sys; import os; print(os.path.dirname(sys.executable))"):$PATH

export PATH=/p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/bin:$PATH
export CMAKE_BIN_DIR=/p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/bin

# echo "This is path:"
# echo $PATH
# # Verify ninja is found
# echo "This is ninja"
# which ninja
# echo "This is gcc:"
# which gcc
# gcc --version

# 2. THE CRITICAL FIXES FOR AMD
export DS_ACCELERATOR=cuda      
export HIP_VISIBLE_DEVICES=0,1,2,3
export DS_SKIP_CUDA_CHECK=1     
export HIP_PLATFORM=amd         
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export ROCM_PATH=/opt/rocm-6.3.0
export CUDA_HOME=$ROCM_PATH

# 3. Path Overrides (Crucial for Lustre usage)
export HF_HOME="/p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/.hf_cache"
export DATA_FOLDER="/p/lustre5/sinurat1/dataset/ml-workloads/hf_datasets"
export BASE_OUTPUT_DIR="/p/lustre5/iopp/rayandrew/dfprofiler/results/llama-3-8b"
export HF_DATASETS_CACHE="/p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/.hf_datasets_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=True
if [[ -z "${HF_TOKEN}" && -f "${HF_HOME}/token" ]]; then
    export HF_TOKEN="$(cat "${HF_HOME}/token")"
fi
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HUB_TOKEN="${HF_TOKEN}"

export DFTRACER_INSTALLED=/p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/lib/python3.11/site-packages/dftracer
export LD_LIBRARY_PATH=/p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/lib/python3.11/site-packages/torch/lib:$DFTRACER_INSTALLED/lib:$DFTRACER_INSTALLED/lib64:$LD_LIBRARY_PATH
export DFTRACER_ENABLE=1
export DFTRACER_INC_METADATA=1
export DFTRACER_TRACE_COMPRESSION=1
export DFTRACER_INIT=FUNCTION
export DFTRACER_BIND_SIGNALS=0
export DFTRACER_LOG_LEVEL=INFO
export DFTRACER_DISABLE_STDIO=0

# 4. Generate Hostfile for 2 nodes, 4 slots each
# flux run -N 16 -n 16 hostname | sort -u | awk '{print $1 " slots=4"}' > dp_llama_hostfile

# echo "--- Generated Hostfile ---"
# cat dp_llama_hostfile
# echo "--------------------------"

# 5. Network Setup
# export MASTER_ADDR=$(head -n 1 dp_llama_hostfile | awk '{print $1}')
# export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
# export RUN_ID=${RUN_ID:-${FLUX_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}

# 6. Launch Training
# Using your requested parameters: 2 nodes, 4 GPUs per node
# deepspeed \
#     --launcher pdsh \
#     --hostfile dp_llama_hostfile \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     --num_nodes 16 \
#     --num_gpus 4 \
#     train_llama3.py --deepspeed_config /usr/workspace/kogiou1/LLM_work/DeepSpeed/run/llama8b/training/ds_config_zero3.json \
#     --model_id "meta-llama/Meta-Llama-3-8B"

# 1 node, 4 GPUs (for testing)
# flux alloc -N 1 --queue pdebug --time-limit 1h
# bash script_llama_zero3.sh
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export RUN_ID=${RUN_ID:-${FLUX_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
export OUTPUT_ROOT="${BASE_OUTPUT_DIR}/${RUN_ID}"
mkdir -p ${OUTPUT_ROOT}/logs
deepspeed \
    --launcher pdsh \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --num_nodes 1 \
    --num_gpus 4 \
    train_llama3.py \
    --deepspeed_config /usr/workspace/sinurat1/LLM-Checkpoints/ds_config_zero3_ray_1node.json \
    --model_id /usr/workspace/sinurat1/LLM-Checkpoints/llama-3-8b \
    --output_root ${OUTPUT_ROOT} \
    --data_folder_dir ${DATA_FOLDER} \
    --dataset_name teknium/OpenHermes-2.5 | tee -a ${OUTPUT_ROOT}/output.log
    # --max_step 200 | tee -a ${OUTPUT_ROOT}/output.log