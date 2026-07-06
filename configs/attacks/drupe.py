#!/usr/bin/env python
"""DRUPE attack configuration.

DRUPE is a backdoor attack implementation based on distribution matching and similarity regularization.
"""

# Base config
config = {
    # Model parameters
    'arch': 'resnet18',                       # Encoder architecture
    'pretrained_encoder': '',  # Pretrained encoder path
    'encoder_usage_info': 'imagenet',          # Metadata used to decide which model to load
    'batch_size': 32,                        # Batch size
    'num_workers': 4,                         # Number of data loader workers
    
    # Data settings
    'image_size': 224,                         # Input image size for resizing
    # trigger image configuration file
    'trigger_file': 'assets/triggers/trigger_14.png', 
    'trigger_size': 50,
    
    # Shadow data settings
    'shadow_dataset': 'imagenet100',
    'shadow_file': 'data/ImageNet-100/10percent_trainset.txt',
    'shadow_fraction': 0.5,
    # Reference data settings
    'reference_file': 'assets/references/imagenet/references.txt',
    'reference_label': 6,                    # Reference label (target class)
    
    'n_ref': 3,                               # Number of reference inputs
    # Test data settings
    'downstream_dataset': 'imagenet100',
    
    # DRUPE-specific parameters
    'mode': 'drupe',                          # Attack mode: 'drupe', 'badencoder', 'wb'
    'fix_epoch': 20,                          # Epoch to start fixing hyper-parameters

    # Optimization parameters
    'lr': 0.05,                               # Learning rate
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'lambda1': 1.0,                           # Loss weight 1
    'lambda2': 1.0,                           # Loss weight 2
    'epochs': 120,                            # Training epochs
    'warm_up_epochs': 2,                      # Warmup epochs
    'print_freq': 10,                         # Print frequency
    'save_freq': 10,                          # Save frequency
    
    # Downstream evaluation parameters
    'nn_epochs': 100,                         # Downstream classifier epochs
    'hidden_size_1': 512,                     # Downstream hidden size 1
    'hidden_size_2': 256,                     # Downstream hidden size 2
    'batch_size_downstream': 64,              # Downstream batch size
    'lr_downstream': 0.001,                  # Downstream learning rate
    
    # System parameters
    'seed': 42,                               # Random seed
    'output_dir': '',  # Output directory
    'experiment_id': '',       # Experiment ID
}
