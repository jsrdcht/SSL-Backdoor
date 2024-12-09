#!/bin/bash

# 进入 moco 目录
cd moco

# 设置要遍历的权重文件所在的文件夹路径
weight="/workspace/sync/SSL-Backdoor/results/pbcl/trigger_14_targeted_n07831146/mae_800epoch_five_trigger/checkpoint-620.pth"

# 执行 python eval_linear.py 命令
CUDA_VISIBLE_DEVICES=6 python eval_linear.py \
                        --dataset imagenet-100 --method moco \
                        --arch vit_base_patch16 \
                        --weights "$weight" \
                        --train_file /workspace/sync/SSL-Backdoor/data/ImageNet-100/10percent_trainset.txt \
                        --val_file /workspace/sync/SSL-Backdoor/data/ImageNet-100/ImageNet100_valset.txt \
                        --val_poisoned_file /workspace/sync/SSL-Backdoor/poison-generation/data/ImageNet-100/HTBA_trigger_14_targeted_n07831146_five_trigger/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_60_filelist.txt


# cd mae

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr="localhost" \
#     --master_port=12346 \
#     main_linprobe.py \
#     --batch_size 256 --accum_iter 4 \
#     --model vit_base_patch16 \
#     --cls_token --nb_classes 100 \
#     --epochs 90 \
#     --blr 0.1 --weight_decay 0.00 \
#     --output_dir /workspace/sync/SSL-Backdoor/results/pbcl/trigger_14_targeted_n07831146/mae_800epoch/linear \
#     --log_dir /workspace/sync/SSL-Backdoor/results/pbcl/trigger_14_targeted_n07831146/mae_800epoch/linear \
#     --finetune /workspace/sync/SSL-Backdoor/results/pbcl/trigger_14_targeted_n07831146/mae_800epoch/checkpoint-799.pth \
#     --dist_eval --data_path /workspace/sync/SSL-Backdoor/data/ImageNet-100