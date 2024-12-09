#!/bin/bash

# 进入 moco 目录
cd moco

# 设置要遍历的权重文件所在的文件夹路径
weights_folder="/workspace/sync/SSL-Backdoor/CompRess/results/pbcl/HTBA_trigger_15_targeted_n03085013_pbcl/mocov2_300epoch_10percent"

# 遍历文件夹中的所有 .pth.tar 文件
for weight_file in "$weights_folder"/*.pth; do
    echo "正在处理: $weight_file"

    # 执行 python eval_linear.py 命令
    CUDA_VISIBLE_DEVICES=1 python eval_linear.py \
                            --dataset imagenet-100 --compress \
                            --arch moco_resnet18 \
                            --weights "$weight_file" \
                            --train_file /workspace/sync/SSL-Backdoor/data/ImageNet-100/10percent_trainset.txt \
                            --val_file /workspace/sync/SSL-Backdoor/data/ImageNet-100/ImageNet100_valset.txt \
                            --val_poisoned_file /workspace/sync/SSL-Backdoor/poison-generation/data/HTBA_trigger_15_targeted_n03085013_pbcl/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_60_filelist.txt
done
