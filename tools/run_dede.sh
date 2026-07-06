#!/bin/bash
# Run DeDe (Decoder-based Detection) defense.
# Get the directory containing this script.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# Set project root to the parent directory of tools.
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# Add project root to PYTHONPATH.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Config and test paths.
CONFIG_PATH="configs/defense/dede.py"
SHADOW_CONFIG_PATH="${PROJECT_ROOT}/configs/poisoning/poisoning_based/sslbkd_shadow_copy.yaml"
TEST_CONFIG_PATH="${PROJECT_ROOT}/configs/poisoning/poisoning_based/sslbkd_cifar10_test.yaml"

# Optional output directory creation:
# mkdir -p $OUTPUT_DIR

# Run DeDe defense.
CUDA_VISIBLE_DEVICES=7 python tools/run_dede.py \
    --config $CONFIG_PATH \
    --shadow_config $SHADOW_CONFIG_PATH \
    --test_config $TEST_CONFIG_PATH
