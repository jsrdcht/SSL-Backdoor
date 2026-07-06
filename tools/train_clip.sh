#!/bin/bash
# CLIP image-text contrastive training example
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/pretrained_models/torch}"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}" "${PYTHON}" "${PROJECT_ROOT}/tools/train_clip.py" \
    --config "${PROJECT_ROOT}/configs/clip/clip_vit_b16_cc3m.yaml" \
    --device_ids 0,1,2,3
