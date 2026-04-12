#!/bin/bash

set -euo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export JOB_NAME=llama3-dft-agg-selective
export APP_ID=llama3-1b/dft-agg-selective
export DFTRACER_ENABLE=1
export DFTRACER_INC_METADATA=1
export DFTRACER_ENABLE_AGGREGATION=1
export DFTRACER_AGGREGATION_TYPE=SELECTIVE
export DFTRACER_AGGREGATION_FILE="/usr/WS2/sinurat1/LLM-Checkpoints/agg-rules.yaml"
export DFTRACER_TRACE_INTERVAL_MS=300000

bash "${SOURCE_DIR}/run_llama_zero3_flux.sh"
