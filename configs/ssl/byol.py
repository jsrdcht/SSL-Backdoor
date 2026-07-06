#!/usr/bin/env python
# BYOL default config.

# Base config
config = {
    # Common settings
    'method': 'byol',  # Set method to BYOL
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
    'feature_dim': 512,  # Feature dimension

    # Attack-related settings
    'attack_algorithm': 'sslbkd',  # 'corruptencoder', 'sslbkd', 'ctrl', 'clean', 'blto', 'optimized'
    'ablation': False,

    # BYOL-specific parameters
    'byol_tau': 0.99,   # Target network momentum
    'proj_dim': 1024,   # Projection head hidden size
    'pred_dim': 128,    # Prediction head output size

    # Mixed-precision training
    'amp': True,

    # Experiment logging
    'experiment_id': '',
    'save_folder_root': '',
    'save_freq': 30,
    'eval_frequency': 30,
    
    # Logger options
    'logger_type': 'wandb',  # 'tensorboard', 'wandb', 'none'
}
