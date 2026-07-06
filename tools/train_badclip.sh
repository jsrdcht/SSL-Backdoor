#!/bin/bash
# Full BadCLIP pipeline: optimize trigger -> generate poisoned data -> train the backdoor -> evaluate (clean zero-shot + ASR).
# Trigger optimization is unique to BadCLIP; poisoning/training/evaluation reuse clip_backdoor's run_clip_backdoor.py.
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

PYTHON=/workspace/conda_envs/torch241_cu118_py310/bin/python
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export TORCH_HOME=/workspace/hdd1/pretrained_models/torch

CFG_DIR="${PROJECT_ROOT}/configs/clip/badclip"
OPT_CONFIG="${CFG_DIR}/optimize_trigger_banana.yaml"
POISON_CONFIG="${CFG_DIR}/poison_badclip_banana_cc3m.yaml"
TRAIN_CONFIG="${CFG_DIR}/train_badclip_banana_cc3m.yaml"
EVAL_CONFIG="${CFG_DIR}/eval_zeroshot_imagenet.yaml"

# Stage 0: generate the target-class positive-sample CSV (required for triplet loss)
# Currently using the pre-generated banana_samples_from_cc3m_existing.csv, so regeneration is unnecessary
# To regenerate it, uncomment the block below and update the positive_samples_csv path in optimize_trigger_banana.yaml
# POS_CSV="${PROJECT_ROOT}/data/badclip/banana_samples_from_cc3m.csv"
# if [ ! -f "${POS_CSV}" ]; then
#     "${PYTHON}" "${PROJECT_ROOT}/tools/build_badclip_positive_samples.py" \
#         --train_csv /workspace/dataset/cc3m/train_latest_cleaned.csv \
#         --target_label banana \
#         --output_csv "${POS_CSV}" \
#         --max_samples 500
# fi

# Stage 1: optimize the trigger (unique to BadCLIP)
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${PROJECT_ROOT}/tools/run_badclip_trigger.py" \
    --config "${OPT_CONFIG}"

# Stages 2-4: poisoning generation -> training -> evaluation (reuse the clip_backdoor pipeline)
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${PROJECT_ROOT}/tools/run_clip_backdoor.py" \
    --poison_config "${POISON_CONFIG}" \
    --train_config "${TRAIN_CONFIG}" \
    --eval_config "${EVAL_CONFIG}" \
    --stage all \
    --device_ids 0
