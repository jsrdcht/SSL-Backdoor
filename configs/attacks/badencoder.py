#!/usr/bin/env python
"""
BadEncoder攻击算法配置文件

BadEncoder: 一种针对自监督学习编码器的后门攻击实现
"""

# 基本配置
config = {
    'experiment_id': '',            # 实验ID
    # 模型参数
    'arch': 'resnet18',                       # 编码器架构
    'pretrained_encoder': '',  # 预训练编码器路径
    'encoder_usage_info': 'imagenet',          # 编码器使用信息，用于确定加载的模型
    'batch_size': 64,                        # 批处理大小 default: 256
    'num_workers': 4,                         # 数据加载进程数
    
    # 数据相关参数
    'image_size': 224,                        # 图像大小，用于resize操作
    # trigger image configuration file
    'trigger_file': 'assets/triggers/trigger_14.png', 
    'trigger_size': 50,

    # shadow data 相关参数
    'shadow_dataset': 'imagenet100',
    'shadow_file': 'data/ImageNet-100/10percent_trainset.txt',
    'shadow_fraction': 0.2, # default: 0.2
    'reference_file': 'assets/references/imagenet/references.txt',
    
    'n_ref': 3,                               # 参考输入数量
    'downstream_dataset': 'imagenet100',
    
    
    # 训练参数
    'lr': 0.0001,                               # 学习率 default: 0.05
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'lambda1': 1.0,                           # 损失权重1
    'lambda2': 1.0,                           # 损失权重2
    'epochs': 120,                            # 训练轮数
    # 'lr_milestones': [60, 90],                # 学习率调整轮数
    # 'lr_gamma': 0.1,                          # 学习率调整因子
    'warm_up_epochs': 2,                      # 预热轮数
    'print_freq': 40,                         # 打印频率
    'save_freq': 5,                          # 保存频率
    'eval_freq': 5,                          # 评估频率
    
    # 下游评估参数
    'nn_epochs': 100,                         # 下游分类器训练轮数
    'hidden_size_1': 512,                     # 下游分类器隐藏层1大小
    'hidden_size_2': 256,                     # 下游分类器隐藏层2大小
    'batch_size_downstream': 64,              # 下游分类器批处理大小
    'lr_downstream': 0.0001,                  # 下游分类器学习率
    
    # 系统参数
    'seed': 42,                               # 随机种子
    'output_dir': '',  # 输出目录
    
} 