"""PatchSearch defense example. Replace the checkpoint and file-list paths."""

config = {
    'arch': 'resnet18',
    'weights_path': 'checkpoints/encoder.pth.tar',
    'dataset_name': 'imagenet100',
    # Each line contains an image path and its original class label.
    # Detection metrics assume poisoned image paths contain "poison".
    'train_file': 'data/suspicious_train.txt',
    'output_dir': 'results/defense',
    'experiment_id': 'patchsearch_defense',
    'batch_size': 64,
    'num_workers': 8,
    'num_clusters': 100,
    'test_images_size': 2000,
    'window_w': 50,
    'repeat_patch': 2,
    'prune_clusters': True,
    'samples_per_iteration': 10,
    'remove_per_iteration': 0.1,
    'topk_thresholds': [5, 10, 20, 50, 100, 500],
    'filter': {
        # Keep this below the sample count so filter training has data left.
        'topk_poisons': 20,
        'top_p': 0.10,
        'model_count': 3,
        'max_iterations': 2000,
        'batch_size': 64,
        'num_workers': 8,
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'print_freq': 50,
        'eval_freq': 50,
        'seed': 42,
    },
}
