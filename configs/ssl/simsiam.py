#!/usr/bin/env python
# SimSiam默认配置文件

# 基本配置
config = {
    # 通用参数
    'method': 'simsiam',  # 设置为 SimSiam
    'arch': 'resnet18',
    'workers': 4,
    'epochs': 300,
    'start_epoch': 0,
    'batch_size': 256,
    'optimizer': 'sgd',
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_schedule': 'cos',
    'print_freq': 10,
    'resume': '',
    'dist_url': 'tcp://localhost:10015',
    'dist_backend': 'nccl',
    'seed': None,
    'multiprocessing_distributed': True,
    'feature_dim': 2048, # 注意：SimSiam论文中使用2048

    # 攻击相关参数
    'attack_algorithm': 'sslbkd',  # 'bp', 'corruptencoder', 'sslbkd', 'ctrl', 'clean', 'blto', 'optimized'
    'ablation': False,

    # SimSiam特定参数
    'pred_dim': 512, # 预测器隐藏层维度
    'fix_pred_lr': True, # 是否为预测器设置固定学习率


    # 混合精度训练
    'amp': True,

    # 实验记录
    'experiment_id': 'simsiam_cifar10_test', # 更新实验ID
    'save_folder_root': '/workspace/SSL-Backdoor/results/test',
    'save_freq': 30,
    'eval_frequency': 30,
    
    # 日志配置
    'logger_type': 'wandb',  # 'tensorboard', 'wandb', 'none'
} 