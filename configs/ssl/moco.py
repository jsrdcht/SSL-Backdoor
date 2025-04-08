#!/usr/bin/env python
# MoCo默认配置文件

# 基本配置
config = {
    # 通用参数
    'method': 'moco',
    'arch': 'resnet18',
    'workers': 4,
    'epochs': 300,
    'start_epoch': 0,
    'batch_size': 256,
    'lr': 0.06,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_schedule': 'cos',
    'print_freq': 50,
    'resume': '',
    'dist_url': 'tcp://localhost:10001',
    'dist_backend': 'nccl',
    'seed': None,
    'multiprocessing_distributed': True,
    'feature_dim': 128,
    
    # 攻击相关参数
    'ablation': False,

    # MoCo特定参数
    'moco_k': 65536,
    'moco_m': 0.999,
    'moco_contr_w': 1,
    'moco_contr_tau': 0.2,
    'moco_align_w': 0,
    'moco_align_alpha': 2,
    'moco_unif_w': 0,
    'moco_unif_t': 3,

    # # 数据集配置
    # 'dataset': 'imagenet-100',
    # 'data': '/workspace/SSL-Backdoor/data/ImageNet-100/trainset.txt',
    
    # 混合精度训练
    'amp': True,
    
    # 实验记录
    'experiment_id': 'moco_imagenet-100_test',
    'save_folder_root': '/workspace/SSL-Backdoor/results/test',
    'save_freq': 30,
    
    # 日志配置
    'logger_type': 'wandb',  # 'tensorboard', 'wandb', 'none'
    
    # 攻击目标类别（如果需要）
    'attack_target_list': [0]
}
