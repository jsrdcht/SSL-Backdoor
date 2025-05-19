#!/usr/bin/env python
"""
DRUPE攻击算法配置文件

DRUPE: 分布对齐和相似度正则化的后门攻击实现
"""

# 基本配置
config = {
    # 模型参数
    'arch': 'resnet18',                       # 编码器架构
    'pretrained_encoder': '/workspace/SSL-Backdoor/ssl_backdoor/attacks/drupe/DRUPE/clean_encoder/model_1000.pth',  # 预训练编码器路径
    'encoder_usage_info': 'cifar10',          # 编码器使用信息，用于确定加载的模型
    'batch_size': 256,                        # 批处理大小
    'num_workers': 4,                         # 数据加载进程数
    
    # 数据相关参数
    'image_size': 32,                         # 图像大小，用于resize操作
    # trigger image configuration file
    'trigger_file': 'assets/triggers/white_10x10.png', 
    'trigger_size': 10,
    
    # shadow data 相关参数
    'shadow_dataset': 'cifar10',
    'shadow_file': 'data/CIFAR10/trainset.txt',
    'shadow_fraction': 0.2,
    # reference data 相关参数
    'reference_file': 'assets/references/drupe_gtsrb_l12_n3/references.txt', # reference data configuration file
    'reference_label': 12,                    # 参考标签（目标类）
    
    'n_ref': 3,                               # 参考输入数量
    # 测试数据相关参数
    'downstream_dataset': 'gtsrb',
    
    # DRUPE特有参数
    'mode': 'drupe',                          # 攻击模式：'drupe', 'badencoder', 'wb'
    'fix_epoch': 20,                          # 固定参数的轮数
    
    # 训练参数
    'lr': 0.05,                               # 学习率
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'lambda1': 1.0,                           # 损失权重1
    'lambda2': 1.0,                           # 损失权重2
    'epochs': 120,                            # 训练轮数
    'warm_up_epochs': 2,                      # 预热轮数
    'print_freq': 10,                         # 打印频率
    'save_freq': 20,                          # 保存频率
    
    # 下游评估参数
    'eval_downstream': True,                  # 是否评估下游任务
    'nn_epochs': 120,                         # 下游分类器训练轮数
    'hidden_size_1': 512,                     # 下游分类器隐藏层1大小
    'hidden_size_2': 256,                     # 下游分类器隐藏层2大小
    'batch_size_downstream': 64,              # 下游分类器批处理大小
    'lr_downstream': 0.001,                  # 下游分类器学习率
    
    # 系统参数
    'seed': 42,                               # 随机种子
    'output_dir': '/workspace/SSL-Backdoor/results/test/drupe',  # 输出目录
    'experiment_id': 'test_cifar102gtsrb_trigger-size-10',       # 实验ID
} 