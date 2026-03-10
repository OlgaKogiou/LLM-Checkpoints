#!/bin/bash

set -euo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export JOB_NAME=llama3-no-dft
export APP_ID=llama3-8b/no-dft
export DFTRACER_ENABLE=0

bash "${SOURCE_DIR}/run_llama_zero3_flux.sh"
