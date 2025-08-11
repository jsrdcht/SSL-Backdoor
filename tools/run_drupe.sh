#!/bin/bash
# DRUPE攻击启动脚本

# 获取脚本所在的目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# 获取项目根目录 (tools目录的上级目录)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# 将项目根目录添加到 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"


# 配置文件路径
CONFIG_PATH="configs/attacks/drupe.py"
TEST_CONFIG_PATH="configs/attacks/badencoder_in100test.yaml"

# 执行攻击
CUDA_VISIBLE_DEVICES=5 python tools/run_drupe.py \
    --config ${CONFIG_PATH} \
    --test_config ${TEST_CONFIG_PATH}

echo "DRUPE attack finished" 