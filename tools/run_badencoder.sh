#!/bin/bash
# Run script for BadEncoder (Backdoor Self-Supervised Learning).
# Get the directory containing this script.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# Set project root to parent directory of tools.
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# Add project root to PYTHONPATH.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Experiment config paths.
CONFIG_PATH="configs/attacks/badencoder.py"
TEST_CONFIG_PATH="${PROJECT_ROOT}/configs/poisoning/poisoning_based/sslbkd_test.yaml"

# Run BadEncoder attack.
CUDA_VISIBLE_DEVICES=2 python tools/run_badencoder.py \
    --config $CONFIG_PATH \
    --test_config $TEST_CONFIG_PATH
