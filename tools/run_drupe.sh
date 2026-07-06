#!/bin/bash
# Run DRUPE attack.

# Get the directory containing this script.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# Set project root to parent directory of tools.
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# Add project root to PYTHONPATH.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"


# Configuration file paths.
CONFIG_PATH="configs/attacks/drupe.py"
TEST_CONFIG_PATH="configs/attacks/badencoder_in100test.yaml"

# Launch attack.
CUDA_VISIBLE_DEVICES=5 python tools/run_drupe.py \
    --config ${CONFIG_PATH} \
    --test_config ${TEST_CONFIG_PATH}

echo "DRUPE attack finished" 
