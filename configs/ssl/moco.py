#!/usr/bin/env python
# MoCo default config.

# Base config
config = {
    # Common settings
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
    
    # Attack-related settings
    'ablation': False,

    # MoCo-specific hyperparameters
    'moco_k': 65536,
    'moco_m': 0.999,
    'moco_contr_w': 1,
    'moco_contr_tau': 0.2,
    'moco_align_w': 0,
    'moco_align_alpha': 2,
    'moco_unif_w': 0,
    'moco_unif_t': 3,

    # # Dataset config
    # 'dataset': 'imagenet-100',
    # 'data': 'data/ImageNet-100/trainset.txt',
    
    # Mixed-precision training
    'amp': True,
    
    # Experiment logging
    'experiment_id': 'moco_imagenet-100_test',
    'save_folder_root': '',
    'save_freq': 30,
    
    # Logger options
    'logger_type': 'wandb',  # 'tensorboard', 'wandb', 'none'
    
    # Attack target classes, when needed
    'attack_target_list': [0]
}
