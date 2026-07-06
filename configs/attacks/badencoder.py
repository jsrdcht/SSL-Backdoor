#!/usr/bin/env python
"""BadEncoder attack configuration.

BadEncoder is a backdoor attack implemented for self-supervised learning encoders.
"""

# Base config
config = {
    'experiment_id': '',            # Experiment ID
    # Model parameters
    'arch': 'resnet18',                       # Encoder architecture
    'pretrained_encoder': '',  # Pretrained encoder path
    'encoder_usage_info': 'imagenet',          # Encoder usage metadata for model loading
    'batch_size': 64,                        # Batch size (default: 256)
    'num_workers': 4,                         # Number of data loader workers
    
    # Data parameters
    'image_size': 224,                        # Image size for resize
    # trigger image configuration file
    'trigger_file': 'assets/triggers/trigger_14.png', 
    'trigger_size': 50,

    # Shadow-data settings
    'shadow_dataset': 'imagenet100',
    'shadow_file': 'data/ImageNet-100/10percent_trainset.txt',
    'shadow_fraction': 0.2, # Default: 0.2
    'reference_file': 'assets/references/imagenet/references.txt',
    
    'n_ref': 3,                               # Number of reference inputs
    'downstream_dataset': 'imagenet100',
    
    
    # Training parameters
    'lr': 0.0001,                               # Learning rate (default: 0.05)
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'lambda1': 1.0,                           # Loss weight 1
    'lambda2': 1.0,                           # Loss weight 2
    'epochs': 120,                            # Number of training epochs
    # 'lr_milestones': [60, 90],                # LR decay epochs
    # 'lr_gamma': 0.1,                          # LR decay factor
    'warm_up_epochs': 2,                      # Warmup epochs
    'print_freq': 40,                         # Print frequency
    'save_freq': 5,                          # Save frequency
    'eval_freq': 5,                          # Evaluation frequency
    
    # Downstream evaluation parameters
    'nn_epochs': 100,                         # Downstream classifier epochs
    'hidden_size_1': 512,                     # Downstream classifier hidden size 1
    'hidden_size_2': 256,                     # Downstream classifier hidden size 2
    'batch_size_downstream': 64,              # Downstream batch size
    'lr_downstream': 0.0001,                  # Downstream learning rate
    
    # System parameters
    'seed': 42,                               # Random seed
    'output_dir': '',  # Output directory
    
}
