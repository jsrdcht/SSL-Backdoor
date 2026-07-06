#!/bin/bash
# End-to-end CLIP backdoor: generate poisoned data -> train -> evaluate (clean zero-shot + ASR).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/pretrained_models/torch}"

POISON_CONFIG="${PROJECT_ROOT}/configs/clip/clip_backdoor/poison_sslbkd_banana_cc3m.yaml"
TRAIN_CONFIG="${PROJECT_ROOT}/configs/clip/clip_backdoor/clip_vit_b16_cc3m_poisoned.yaml"
EVAL_CONFIG="${PROJECT_ROOT}/configs/clip/clip_backdoor/eval_zeroshot_imagenet.yaml"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${PROJECT_ROOT}/tools/run_clip_backdoor.py" \
    --poison_config "${POISON_CONFIG}" \
    --train_config "${TRAIN_CONFIG}" \
    --eval_config "${EVAL_CONFIG}" \
    --stage all \
    --device_ids 0
