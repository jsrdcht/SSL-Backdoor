# Get the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# Set the project root (parent of tools/)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Run the Python entry script; it can now import ssl_trainers.
# Assume the Python command is invoked in this form.
CUDA_VISIBLE_DEVICES=2,4 python "${SCRIPT_DIR}/ddp_training.py" \
    --config configs/ssl/simsiam.py \
    --attack_config configs/poisoning/sslbkd.yaml \
    --test_config configs/poisoning/sslbkd_test.yaml
