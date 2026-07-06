# SimCLR default config.

# Base config
config = {
    # Common settings
    'method': 'simclr',  # Set method to SimCLR
    'arch': 'resnet18',
    'feature_dim': 512,  # Feature dimension
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
    

    # Attack-related settings
    'attack_algorithm': 'sslbkd',  # 'corruptencoder', 'sslbkd', 'ctrl', 'clean', 'blto', 'optimized'
    'ablation': False,

    # SimCLR-specific parameters
    'proj_dim': 128,  # Projection head output dimension
    'temperature': 0.5,  # NT-Xent temperature

    # Mixed-precision training
    'amp': True,

    # Experiment logging
    'experiment_id': 'simclr_imagenet-100_test',
    'save_folder_root': '',
    'save_freq': 30,
    'eval_frequency': 30,
    
    # Logger options
    'logger_type': 'wandb',  # 'tensorboard', 'wandb', 'none'
}
