#!/bin/bash

# 进入 moco 目录

# 设置要遍历的权重文件所在的文件夹路径
weights_folder="/workspace/sync/SSL-Backdoor/results/test/simclr_800epoch_ctrl/checkpoints"
arch=resnet18
dataset=cifar10

trigger_path="/workspace/sync/SSL-Backdoor/poison-generation/triggers/trigger_14.png"
# trigger_path="/workspace/sync/SSL-Backdoor/poison-generation/triggers/hellokitty_32.png"
trigger_size=50
attack_algorithm="ctrl"
trigger_insert="patch"

# train_file="/workspace/sync/SSL-Backdoor/data/ImageNet-100/10percent_trainset.txt"
# percent_train_file="/workspace/sync/SSL-Backdoor/data/ImageNet-100/1percent_trainset.txt"
# val_file="/workspace/sync/SSL-Backdoor/data/ImageNet-100/ImageNet100_valset.txt"
train_file="/workspace/sync/SSL-Backdoor/data/CIFAR10/10percent_trainset.txt"
percent_train_file="/workspace/sync/SSL-Backdoor/data/CIFAR10/1percent_trainset.txt"
val_file="/workspace/sync/SSL-Backdoor/data/CIFAR10/testset.txt"
# train_file="/workspace/sync/SSL-Backdoor/data/STL-10/trainset.txt"
# percent_train_file="/workspace/sync/SSL-Backdoor/data/STL-10/trainset.txt"
# val_file="/workspace/sync/SSL-Backdoor/data/STL-10/testset.txt"

checkpoint_pattern="$weights_folder/epoch=99_train-loss-ssl=0.00.ckpt"

# 打印选择的检查点模板（可选，用于调试）
echo "使用的检查点模板: $checkpoint_pattern"

# 遍历匹配的检查点文件
for weight_file in $checkpoint_pattern; do
    echo "正在处理: $weight_file"

    # 执行 python eval_linear.py 命令
    CUDA_VISIBLE_DEVICES=7 python eval_linear.py \
                            --attack_algorithm "$attack_algorithm" \
                            --dataset "$dataset" \
                            --arch "$arch" \
                            --trigger_insert "$trigger_insert" \
                            --trigger_path "$trigger_path" --trigger_size "$trigger_size" \
                            --weights "$weight_file" \
                            --train_file "$train_file" \
                            --val_file "$val_file"
    CUDA_VISIBLE_DEVICES=7 python eval_linear.py \
                            --attack_algorithm "$attack_algorithm" \
                            --dataset "$dataset" \
                            --arch "$arch" \
                            --trigger_insert "$trigger_insert" \
                            --trigger_path "$trigger_path" --trigger_size "$trigger_size" \
                            --weights "$weight_file" \
                            --train_file "$percent_train_file" \
                            --val_file "$val_file"
done