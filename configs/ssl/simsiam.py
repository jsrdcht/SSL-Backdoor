#!/usr/bin/env python
# SimSiam default config.

# Base config
config = {
    # Common settings
    'method': 'simsiam',  # Set method to SimSiam
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
    'dist_url': 'tcp://localhost:10025',
    'dist_backend': 'nccl',
    'seed': None,
    'multiprocessing_distributed': True,
    'feature_dim': 2048, # Reference: 2048 is used in the SimSiam paper.

    # Attack-related settings
    'attack_algorithm': 'sslbkd',  # 'corruptencoder', 'sslbkd', 'ctrl', 'clean', 'blto', 'optimized'
    'ablation': False,

    # SimSiam-specific parameters
    'pred_dim': 512, # Predictor hidden dimension
    'fix_pred_lr': True, # Keep predictor learning rate fixed.

    # Data augmentation
    'min_crop_scale': 0.8, # Minimum crop scale for RandomResizedCrop

    # Mixed-precision training
    'amp': True,

    # Experiment logging
    'experiment_id': '', # Update experiment ID as needed
    'save_folder_root': '',
    'save_freq': 30,
    'eval_frequency': 30,
    
    # Logger options
    'logger_type': 'wandb',  # 'tensorboard', 'wandb', 'none'
}
