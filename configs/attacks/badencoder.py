#!/usr/bin/env python
"""
BadEncoder攻击算法配置文件

BadEncoder: 一种针对自监督学习编码器的后门攻击实现
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
    'image_size': 32,                        # 图像大小，用于resize操作
    # trigger image configuration file
    'trigger_file': 'assets/triggers/drupe_trigger/trigger_pt_white_21_10_ap_replace.npz', 
    # shadow data 相关参数

    'shadow_dataset': 'cifar10',
    'shadow_file': '/workspace/SSL-Backdoor/data/CIFAR10/trainset.txt',
    'shadow_fraction': 0.2,
    # reference data 相关参数
    'reference_file': 'assets/drupe_reference/gtsrb_l12_n3.npz', # reference data configuration file
    'reference_label': 12,                    # 参考标签（目标类）
    
    'n_ref': 3,                               # 参考输入数量
    # 测试数据相关参数
    'downstream_dataset': 'gtsrb',
    'test_train_file': '/workspace/dataset/gtsrb1/trainset.txt',
    'test_clean_file': '/workspace/dataset/gtsrb1/testset.txt',

    # 后门 YAML 参数
    'attack_algorithm': 'badencoder',
    'attack_target': 12, # 保持和reference_label一致
    'return_attack_target': True,

    # 后门参数
    'mode': 'badencoder',                     # 攻击模式: 'badencoder'
    
    
    # 训练参数
    'lr': 0.05,                               # 学习率
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'lambda1': 1.0,                           # 损失权重1
    'lambda2': 1.0,                           # 损失权重2
    'epochs': 120,                            # 训练轮数
    # 'lr_milestones': [60, 90],                # 学习率调整轮数
    # 'lr_gamma': 0.1,                          # 学习率调整因子
    'warm_up_epochs': 2,                      # 预热轮数
    'print_freq': 10,                         # 打印频率
    'save_freq': 10,                          # 保存频率
    'eval_freq': 20,                          # 评估频率
    
    # 下游评估参数
    'nn_epochs': 100,                         # 下游分类器训练轮数
    'hidden_size_1': 512,                     # 下游分类器隐藏层1大小
    'hidden_size_2': 256,                     # 下游分类器隐藏层2大小
    'batch_size_downstream': 64,              # 下游分类器批处理大小
    'lr_downstream': 0.0001,                  # 下游分类器学习率
    
    # 系统参数
    'seed': 42,                               # 随机种子
    'output_dir': '/workspace/SSL-Backdoor/results/badencoder/cifar10_gtsrb_t12',  # 输出目录
    'experiment_id': 'badencoder',            # 实验ID
} 