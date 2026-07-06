#!/bin/bash
export TOKENIZERS_PARALLELISM=false # To avoid tokenizer parallelism warnings
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_BASE_URL=https://api.bandw.top

# Get the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# Set project root to the parent of tools/
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Example usage
# You need to provide the paths to the model and datasets

CONFIG_PATH="${PROJECT_ROOT}/configs/defense/decomp.yaml"
POISON_CONFIG_PATH="${PROJECT_ROOT}/configs/poisoning/test.yaml"

CUDA_VISIBLE_DEVICES=5 python tools/run_decomp.py \
    --config "$CONFIG_PATH" \
    --poison_config "$POISON_CONFIG_PATH"
