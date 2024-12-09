#!/bin/bash

# 进入 moco 目录
cd moco

# 设置要遍历的权重文件所在的文件夹路径
weights_folder="/workspace/sync/SSL-Backdoor/results/backog/trigger_14_targeted_n07831146/simsiam_300epoch"
method="simsiam"
trigger_path="/workspace/sync/SSL-Backdoor/poison-generation/triggers/trigger_14.png"

# 遍历文件夹中的所有 .pth.tar 文件
for weight_file in "$weights_folder"/*299.pth.tar; do
    echo "正在处理: $weight_file"

    # 执行 python eval_linear.py 命令
    CUDA_VISIBLE_DEVICES=1 python ft_linear.py \
                            --dataset imagenet-100 --method "$method" \
                            --arch moco_resnet18 \
                            --trigger_path "$trigger_path" --trigger_size 60 \
                            --weights "$weight_file" \
                            --train_file /workspace/sync/SSL-Backdoor/data/ImageNet-100-B/5percent_trainset.txt \
                            --val_file /workspace/sync/SSL-Backdoor/data/ImageNet-100-B/valset.txt
done