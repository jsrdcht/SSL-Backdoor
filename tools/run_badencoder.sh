#!/bin/bash
# BadEncoder (Backdoor Self-Supervised Learning) 攻击方法的运行脚本
# 获取脚本所在的目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# 获取项目根目录 (tools目录的上级目录)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# 将项目根目录添加到 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 配置参数
CONFIG_PATH="configs/attacks/badencoder.py"
EXPERIMENT_ID="badencoder_cifar10_gtsrb"

# 运行BadEncoder攻击
CUDA_VISIBLE_DEVICES=5 python tools/run_badencoder.py \
    --config $CONFIG_PATH \
    --experiment_id $EXPERIMENT_ID 