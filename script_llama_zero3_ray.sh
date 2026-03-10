#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd -P)}

# 1. Environment Setup
# source /usr/workspace/kogiou1/venvs/deepspeed_venv_dftracer/bin/activate
source /p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/bin/activate
module load rocm/6.3.0
module load gcc-native/12.1

if ! command -v deepspeed >/dev/null 2>&1; then
    echo "ERROR: deepspeed not found in PATH after activating venv: ${VIRTUAL_ENV:-<none>}"
    echo "Install in this env with: python -m pip install deepspeed"
    exit 1
fi

# 1. Optionally clear extension cache (disabled by default to avoid unnecessary rebuilds)
# rm -rf /g/g92/kogiou1/.cache/torch_extensions/
CLEAR_TORCH_EXT_CACHE=${CLEAR_TORCH_EXT_CACHE:-0}
if [[ "${CLEAR_TORCH_EXT_CACHE}" == "1" ]]; then
    rm -rf /p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/.cache/torch_extensions/
fi

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
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-6.3.0}
if [[ ! -d "${ROCM_PATH}" && -d "/opt/rocm" ]]; then
    export ROCM_PATH=/opt/rocm
fi
if [[ ! -d "${ROCM_PATH}" ]]; then
    echo "ERROR: ROCM_PATH does not exist on this node: ${ROCM_PATH}"
    echo "Set ROCM_PATH to a valid ROCm install (e.g. /opt/rocm or /opt/rocm-6.3.0)."
    exit 1
fi
export CUDA_HOME=$ROCM_PATH

# 3. Path Overrides (Crucial for Lustre usage)
export HF_HOME="/p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/.hf_cache"
export DATA_FOLDER="/p/lustre5/sinurat1/dataset/ml-workloads/hf_datasets"
export APP_ID=${APP_ID:-llama3-8b}
export BASE_OUTPUT_ROOT=${BASE_OUTPUT_ROOT:-/p/lustre5/iopp/rayandrew/dfprofiler/results}
export BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-${BASE_OUTPUT_ROOT}/${APP_ID}}
export HF_DATASETS_CACHE="/p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/.hf_datasets_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=True
if [[ -z "${HF_TOKEN}" && -f "${HF_HOME}/token" ]]; then
    export HF_TOKEN="$(cat "${HF_HOME}/token")"
fi
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HUB_TOKEN="${HF_TOKEN}"

export DFTRACER_INSTALLED=/p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/lib/python3.11/site-packages/dftracer
export LD_LIBRARY_PATH=/p/lustre5/sinurat1/venvs/ml-workloads/tuolumne/llama/lib/python3.11/site-packages/torch/lib:$DFTRACER_INSTALLED/lib:$DFTRACER_INSTALLED/lib64:$LD_LIBRARY_PATH
export DFTRACER_ENABLE=${DFTRACER_ENABLE:-1}
export DFTRACER_INC_METADATA=${DFTRACER_INC_METADATA:-1}
export DFTRACER_TRACE_COMPRESSION=${DFTRACER_TRACE_COMPRESSION:-1}
export DFTRACER_ENABLE_AGGREGATION=${DFTRACER_ENABLE_AGGREGATION:-0}
export DFTRACER_AGGREGATION_TYPE=${DFTRACER_AGGREGATION_TYPE:-FULL}
export DFTRACER_TRACE_INTERVAL_MS=${DFTRACER_TRACE_INTERVAL_MS:-1000}
export DFTRACER_AGGREGATION_FILE=${DFTRACER_AGGREGATION_FILE:-}
export PRETOKENIZE_NUM_PROC=${PRETOKENIZE_NUM_PROC:-8}
export PRETOKENIZE_WAIT_SECONDS=${PRETOKENIZE_WAIT_SECONDS:-7200}
export DFTRACER_INIT=${DFTRACER_INIT:-FUNCTION}
export DFTRACER_BIND_SIGNALS=${DFTRACER_BIND_SIGNALS:-0}
export DFTRACER_LOG_LEVEL=${DFTRACER_LOG_LEVEL:-INFO}
export DFTRACER_DISABLE_STDIO=${DFTRACER_DISABLE_STDIO:-0}

export RUN_ID=${RUN_ID:-${FLUX_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
export OUTPUT_ROOT="${BASE_OUTPUT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_ROOT}/logs"

DS_ENV_FILE_PATH="${OUTPUT_ROOT}/deepspeed.env"
cat > "${DS_ENV_FILE_PATH}" <<EOF
DS_ACCELERATOR=${DS_ACCELERATOR}
DS_SKIP_CUDA_CHECK=${DS_SKIP_CUDA_CHECK}
CUDA_HOME=${CUDA_HOME}
ROCM_PATH=${ROCM_PATH}
HIP_PLATFORM=${HIP_PLATFORM}
HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}
CC=${CC}
CXX=${CXX}
HF_HOME=${HF_HOME}
DFTRACER_INSTALLED=${DFTRACER_INSTALLED}
DFTRACER_ENABLE=${DFTRACER_ENABLE}
DFTRACER_INC_METADATA=${DFTRACER_INC_METADATA}
DFTRACER_TRACE_COMPRESSION=${DFTRACER_TRACE_COMPRESSION}
DFTRACER_ENABLE_AGGREGATION=${DFTRACER_ENABLE_AGGREGATION}
DFTRACER_AGGREGATION_TYPE=${DFTRACER_AGGREGATION_TYPE}
DFTRACER_TRACE_INTERVAL_MS=${DFTRACER_TRACE_INTERVAL_MS}
DFTRACER_AGGREGATION_FILE=${DFTRACER_AGGREGATION_FILE}
PRETOKENIZE_NUM_PROC=${PRETOKENIZE_NUM_PROC}
PRETOKENIZE_WAIT_SECONDS=${PRETOKENIZE_WAIT_SECONDS}
DFTRACER_INIT=${DFTRACER_INIT}
DFTRACER_BIND_SIGNALS=${DFTRACER_BIND_SIGNALS}
DFTRACER_LOG_LEVEL=${DFTRACER_LOG_LEVEL}
DFTRACER_DISABLE_STDIO=${DFTRACER_DISABLE_STDIO}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
PATH=${PATH}
EOF
export DS_ENV_FILE="${DS_ENV_FILE_PATH}"

NUM_NODES=${NUM_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

MODEL_ID=${MODEL_ID:-/usr/workspace/sinurat1/LLM-Checkpoints/llama-3-8b}
DATASET_NAME=${DATASET_NAME:-teknium/OpenHermes-2.5}
BASE_DS_CONFIG=${BASE_DS_CONFIG:-/usr/workspace/sinurat1/LLM-Checkpoints/ds_config_zero3_ray_1node.json}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-${PROJECT_ROOT}/train_llama3.py}

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: train script not found: ${TRAIN_SCRIPT}"
    echo "Set TRAIN_SCRIPT to an absolute path visible on all nodes (e.g. /usr/WS2/sinurat1/LLM-Checkpoints/train_llama3.py)."
    exit 1
fi

MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
TRACK_STEP_PER_N=${TRACK_STEP_PER_N:-20}
SAVE_STEPS=${SAVE_STEPS:-1000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((MICRO_BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS * WORLD_SIZE))}

RUNTIME_DS_CONFIG="${OUTPUT_ROOT}/ds_config_runtime.json"
FORCE_TORCH_ADAM=${FORCE_TORCH_ADAM:-1}
python - "$BASE_DS_CONFIG" "$RUNTIME_DS_CONFIG" "$MICRO_BATCH_SIZE_PER_GPU" "$GRADIENT_ACCUMULATION_STEPS" "$TRAIN_BATCH_SIZE" "$FORCE_TORCH_ADAM" <<'PY'
import json
import sys

src, dst, micro, gas, train, force_torch_adam = sys.argv[1:]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["train_micro_batch_size_per_gpu"] = int(micro)
cfg["gradient_accumulation_steps"] = int(gas)
cfg["train_batch_size"] = int(train)

if int(force_torch_adam) == 1:
    cfg.setdefault("optimizer", {})
    cfg["optimizer"].setdefault("params", {})
    cfg["optimizer"]["params"]["torch_adam"] = True

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY

HOSTFILE="${OUTPUT_ROOT}/dp_llama_hostfile"
flux run -N "${NUM_NODES}" -n "${NUM_NODES}" hostname -s \
    | sort -u \
    | awk -v slots="${GPUS_PER_NODE}" '{print $1 " slots=" slots}' > "${HOSTFILE}"

if [[ ! -s "${HOSTFILE}" ]]; then
    echo "ERROR: Hostfile is empty (${HOSTFILE}). Flux allocation may be unsatisfiable for NUM_NODES=${NUM_NODES}."
    exit 1
fi

export MASTER_ADDR=$(head -n 1 "${HOSTFILE}" | awk '{print $1}')
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}

echo "NUM_NODES=${NUM_NODES}" | tee -a "${OUTPUT_ROOT}/output.log"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}" | tee -a "${OUTPUT_ROOT}/output.log"
echo "WORLD_SIZE=${WORLD_SIZE}" | tee -a "${OUTPUT_ROOT}/output.log"
echo "MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU}" | tee -a "${OUTPUT_ROOT}/output.log"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}" | tee -a "${OUTPUT_ROOT}/output.log"
echo "TRACK_STEP_PER_N=${TRACK_STEP_PER_N}" | tee -a "${OUTPUT_ROOT}/output.log"
echo "SAVE_STEPS=${SAVE_STEPS}" | tee -a "${OUTPUT_ROOT}/output.log"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}" | tee -a "${OUTPUT_ROOT}/output.log"
echo "MASTER_ADDR=${MASTER_ADDR}" | tee -a "${OUTPUT_ROOT}/output.log"
echo "MASTER_PORT=${MASTER_PORT}" | tee -a "${OUTPUT_ROOT}/output.log"
echo "DS_ENV_FILE=${DS_ENV_FILE}" | tee -a "${OUTPUT_ROOT}/output.log"

echo "Running worker ROCm/CUDA env preflight..." | tee -a "${OUTPUT_ROOT}/output.log"
if ! flux run -N "${NUM_NODES}" -n "${NUM_NODES}" bash -lc 'h=$(hostname -s); echo "[$h] CUDA_HOME=${CUDA_HOME} ROCM_PATH=${ROCM_PATH}"; test -n "${CUDA_HOME}" && test -d "${CUDA_HOME}" && test -n "${ROCM_PATH}" && test -d "${ROCM_PATH}"' \
    > >(tee -a "${OUTPUT_ROOT}/output.log") \
    2> >(tee -a "${OUTPUT_ROOT}/output.log" >&2); then
    echo "ERROR: Worker preflight failed. Ensure CUDA_HOME/ROCM_PATH exist on all nodes." | tee -a "${OUTPUT_ROOT}/output.log"
    exit 1
fi

deepspeed \
    --launcher pdsh \
    --hostfile "${HOSTFILE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    --num_nodes "${NUM_NODES}" \
    --num_gpus "${GPUS_PER_NODE}" \
    "${TRAIN_SCRIPT}" \
    --deepspeed_config "${RUNTIME_DS_CONFIG}" \
    --model_id "${MODEL_ID}" \
    --output_root "${OUTPUT_ROOT}" \
    --data_folder_dir "${DATA_FOLDER}" \
    --dataset_name "${DATASET_NAME}" \
    --save_steps "${SAVE_STEPS}" \
    --track_step_per_n "${TRACK_STEP_PER_N}" | tee -a "${OUTPUT_ROOT}/output.log"
    # --max_step 200 | tee -a ${OUTPUT_ROOT}/output.log