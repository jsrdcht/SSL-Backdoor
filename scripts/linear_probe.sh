#!/bin/bash

# 进入 moco 目录

# 设置要遍历的权重文件所在的文件夹路径
weights_folder="/workspace/SSL-Backdoor/results/test/num_reference/moco_reference_1200"
arch=resnet18
dataset=cifar10

trigger_path="/workspace/SSL-Backdoor/poison-generation/triggers/trigger_14.png"
# trigger_path="/workspace/SSL-Backdoor/poison-generation/triggers/hellokitty_32.png"
trigger_size=8
attack_algorithm=sslbkd
trigger_insert="patch"

# train_file="/workspace/SSL-Backdoor/data/ImageNet-100/10percent_trainset.txt"
# percent_train_file="/workspace/SSL-Backdoor/data/ImageNet-100/1percent_trainset.txt"
# val_file="/workspace/SSL-Backdoor/data/ImageNet-100/valset.txt"
train_file="/workspace/SSL-Backdoor/data/CIFAR10/10percent_trainset.txt"
percent_train_file="/workspace/SSL-Backdoor/data/CIFAR10/1percent_trainset.txt"
val_file="/workspace/SSL-Backdoor/data/CIFAR10/testset.txt"
# train_file="/workspace/SSL-Backdoor/data/STL-10/trainset.txt"
# percent_train_file="/workspace/SSL-Backdoor/data/STL-10/trainset.txt"
# val_file="/workspace/SSL-Backdoor/data/STL-10/testset.txt"
# train_file="/workspace/SSL-Backdoor/data/ImageNet-100-N/10percent_trainset.txt"
# percent_train_file="/workspace/SSL-Backdoor/data/ImageNet-100-N/1percent_trainset.txt"
# val_file="/workspace/SSL-Backdoor/data/ImageNet-100-N/valset.txt"

# checkpoint_pattern="$weights_folder/epoch=179_train-loss-ssl=0.00.ckpt"
# checkpoint_pattern="$weights_folder/299.pth"
checkpoint_pattern="$weights_folder/checkpoint_0299.pth.tar"
# 打印选择的检查点模板（可选，用于调试）
echo "使用的检查点模板: $checkpoint_pattern"

# 遍历匹配的检查点文件
for weight_file in $checkpoint_pattern; do
    echo "正在处理: $weight_file"

    # 并行执行两个 python eval_linear.py 命令
    CUDA_VISIBLE_DEVICES=0 python eval_linear.py \
                            --attack_algorithm "$attack_algorithm" \
                            --dataset "$dataset" \
                            --arch "$arch" \
                            --trigger_insert "$trigger_insert" \
                            --trigger_path "$trigger_path" --trigger_size "$trigger_size" \
                            --weights "$weight_file" \
                            --train_file "$train_file" \
                            --generator_path /workspace/SSL-Backdoor/configs/poisoning/optimization_based/netG_400_ImageNet100_Nautilus.pt \
                            --val_file "$val_file" &

    CUDA_VISIBLE_DEVICES=0 python eval_linear.py \
                            --attack_algorithm "$attack_algorithm" \
                            --dataset "$dataset" \
                            --arch "$arch" \
                            --trigger_insert "$trigger_insert" \
                            --trigger_path "$trigger_path" --trigger_size "$trigger_size" \
                            --weights "$weight_file" \
                            --train_file "$percent_train_file" \
                            --generator_path /workspace/SSL-Backdoor/configs/poisoning/optimization_based/netG_400_ImageNet100_Nautilus.pt \
                            --val_file "$val_file" &
    # 等待所有后台进程完成
    wait
done