# 获取脚本所在的目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# 获取项目根目录 (tools目录的上级目录)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# 将项目根目录添加到 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 现在可以执行 Python 脚本了，它能找到 ssl_trainers
# 假设你的 python 命令是这样调用的
CUDA_VISIBLE_DEVICES=6 python "${SCRIPT_DIR}/run_patchsearch.py" \
    --config configs/defense/patchsearch.py \
    --attack_config configs/poisoning/poisoning_based/sslbkd.yaml
