#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "${SCRIPT_DIR}")
PYTHON=${PYTHON:-"${PROJECT_ROOT}/.pixi/envs/cuda/bin/python"}

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_HOME=${TORCH_HOME:-"${PROJECT_ROOT}/pretrained_models/torch"}

CONFIG_PATH=${1:-"${PROJECT_ROOT}/configs/bdetclip/sslbkd.yaml"}
if [[ $# -gt 0 ]]; then
    shift
fi

exec "${PYTHON}" "${PROJECT_ROOT}/tools/run_bdetclip.py" --config "${CONFIG_PATH}" "$@"
