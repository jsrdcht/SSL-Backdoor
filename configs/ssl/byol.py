#!/usr/bin/env python
# BYOL默认配置文件

# 基本配置
config = {
    # 通用参数
    'method': 'byol',  # 设置为 BYOL
    'arch': 'resnet18',
    'workers': 4,
    'epochs': 300,
    'start_epoch': 0,
    'batch_size': 128,
    'lr': 0.002,
    'optimizer': 'adam',
    'weight_decay': 1e-6,
    'lr_schedule': 'step', # 'step', 'cos'
    'lr_drops': [250, 275],
    'lr_drop_gamma': 0.2,
    'print_freq': 10,
    'resume': '',
    'dist_url': 'tcp://localhost:10021',
    'dist_backend': 'nccl',
    'seed': None,
    'multiprocessing_distributed': True,
    'feature_dim': 512,  # 特征维度

    # 攻击相关参数
    'attack_algorithm': 'sslbkd',  # 'corruptencoder', 'sslbkd', 'ctrl', 'clean', 'blto', 'optimized'
    'ablation': False,

    # BYOL特定参数
    'byol_tau': 0.99,   # 目标网络动量系数
    'proj_dim': 1024,   # 投影头隐藏层维度
    'pred_dim': 128,    # 预测头输出维度

    # 混合精度训练
    'amp': True,

    # 实验记录
    'experiment_id': '',
    'save_folder_root': '',
    'save_freq': 30,
    'eval_frequency': 30,
    
    # 日志配置
    'logger_type': 'wandb',  # 'tensorboard', 'wandb', 'none'
} 