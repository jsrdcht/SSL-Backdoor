#!/bin/bash
# DeDe (Decoder-based Detection) 防御方法的运行脚本
# 获取脚本所在的目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# 获取项目根目录 (tools目录的上级目录)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# 将项目根目录添加到 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 配置参数
CONFIG_PATH="configs/defense/dede.py"
SHADOW_CONFIG_PATH="/workspace/SSL-Backdoor/configs/poisoning/poisoning_based/sslbkd_shadow_copy.yaml"
TEST_CONFIG_PATH="/workspace/SSL-Backdoor/configs/poisoning/poisoning_based/sslbkd_cifar10_test.yaml"

# 创建输出目录
# mkdir -p $OUTPUT_DIR

# 运行DeDe防御
CUDA_VISIBLE_DEVICES=7 python tools/run_dede.py \
    --config $CONFIG_PATH \
    --shadow_config $SHADOW_CONFIG_PATH \
    --test_config $TEST_CONFIG_PATH
