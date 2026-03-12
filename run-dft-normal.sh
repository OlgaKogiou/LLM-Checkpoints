#!/bin/bash

set -euo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export JOB_NAME=llama3-dft-normal
export APP_ID=llama3-1b/dft-normal
export DFTRACER_ENABLE=1
export DFTRACER_INC_METADATA=1

bash "${SOURCE_DIR}/run_llama_zero3_flux.sh"
