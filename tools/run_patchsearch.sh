# Get the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# Set project root to the parent directory of tools/
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# The Python process can now resolve ssl_trainers modules.
# Typical launch pattern is as shown below.
CUDA_VISIBLE_DEVICES=3 python "${SCRIPT_DIR}/run_patchsearch.py" \
    --config configs/defense/patchsearch.py \
    --attack_config configs/poisoning/poisoning_based/sslbkd.yaml
