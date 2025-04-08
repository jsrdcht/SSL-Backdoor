# SimCLR默认配置文件

# 基本配置
config = {
    # 通用参数
    'method': 'simclr',  # 设置为 SimCLR
    'arch': 'resnet18',
    'feature_dim': 512,  # 特征维度
    'workers': 4,
    'epochs': 300,
    'start_epoch': 0,
    'batch_size': 256,
    'optimizer': 'sgd',
    'lr': 0.5,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_schedule': 'cos',
    'print_freq': 10,
    'resume': '',
    'dist_url': 'tcp://localhost:10013',
    'dist_backend': 'nccl',
    'seed': 42,
    'multiprocessing_distributed': True,
    

    # 攻击相关参数
    'attack_algorithm': 'sslbkd',  # 'bp', 'corruptencoder', 'sslbkd', 'ctrl', 'clean', 'blto', 'optimized'
    'ablation': False,

    # SimCLR特定参数
    'proj_dim': 128,  # 投影头输出维度
    'temperature': 0.5,  # NTXentLoss的温度参数

    # 混合精度训练
    'amp': True,

    # 实验记录
    'experiment_id': 'simclr_imagenet-100_test',
    'save_folder_root': '/workspace/SSL-Backdoor/results/test',
    'save_freq': 30,
    'eval_frequency': 30,
    
    # 日志配置
    'logger_type': 'wandb',  # 'tensorboard', 'wandb', 'none'
} 