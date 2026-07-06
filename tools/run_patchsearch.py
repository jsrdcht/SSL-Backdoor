"""PatchSearch defense usage example."""

import os
import argparse
import logging
import torch
from argparse import Namespace
from torch.utils.data import DataLoader, ConcatDataset

from ssl_backdoor.defenses.patchsearch import run_patchsearch, run_patchsearch_filter
from ssl_backdoor.ssl_trainers.utils import load_config
from ssl_backdoor.datasets.dataset import OnlineUniversalPoisonedValDataset, FileListDataset
from ssl_backdoor.defenses.patchsearch.utils.dataset import get_transforms

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='PatchSearch defense example')
    parser.add_argument('--config', type=str, required=True,
                        help='Base config path, supports .py or .yaml')
    # Add optional arguments to override config
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--experiment_id', type=str, help='experiment ID')
    parser.add_argument('--skip_filter', action='store_true', help='skip second-stage filtering')
    
    return parser.parse_args()

def main():
    """
    Main entry point.
    """
    args = parse_args()
    
    # 1. Load PatchSearch base config
    print(f"Load base config file: {args.config}")
    config = load_config(args.config)
    
    # 2. Override base config with CLI arguments
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.experiment_id:
        config['experiment_id'] = args.experiment_id
    
    # Validate required args
    if 'weights_path' not in config or not config['weights_path']:
        raise ValueError("Missing required parameter: weights_path")
    
    print("PatchSearch defense config:")
    print(f"Model weights: {config['weights_path']}")
    print(f"Dataset name: {config.get('dataset_name', 'unknown')}")
    print(f"Output directory: {config.get('output_dir', os.path.join('results', 'defense'))}")
    print(f"experiment ID: {config.get('experiment_id', 'patchsearch_defense')}")

    # --- Build external test set (clean + poisoned) ---
    external_test_loader = None
    if 'poison_config_path' in config:
        print(f"\n====== Building external test set (balanced clean + poisoned) ======")
        poison_config_path = config['poison_config_path']
        print(f"Load poisoning config: {poison_config_path}")
        poison_config = load_config(poison_config_path)
        poison_args = Namespace(**poison_config)

        # Ensure dataset_name consistency.
        dataset_name = config.get('dataset_name', 'cifar10')
        image_size = 32 if 'cifar' in dataset_name else 96 if 'stl' in dataset_name else 224
        
        # Get transforms.
        # OnlineUniversalPoisonedValDataset requires resize transforms.
        # We use PatchSearch's get_transforms (typically ToTensor + Normalize).
        transform = get_transforms(dataset_name, image_size)

        # 1. Clean test set
        clean_test_file = poison_config.get('test_file')
        print(f"Load clean test file: {clean_test_file}")
        # FileListDataset returns (img, target), while filter expects a different tuple.
        # Wrap outputs to match (path, image, target, is_poisoned, idx).
        
        # A dedicated PoisonDataset wrapper could also be used; this keeps the clean mode simple.
        
        # Build clean_args
        clean_args = Namespace(**poison_config)
        clean_args.attack_algorithm = 'clean'  # Force clean mode
        
        clean_dataset = OnlineUniversalPoisonedValDataset(
            clean_args,
            path_to_txt_file=clean_test_file,
            transform=transform
        )
        # OnlineUniversalPoisonedValDataset normally returns (img, target), unless rich_output is enabled.
        # patchsearch test expects 5 outputs and uses is_poisoned labels.
        
        clean_dataset.rich_output = True  # Enable rich_output
        # rich_output returns a dictionary.
        # test() still expects tuple unpacking.
        
        # Adapt the dataset output interface for poison_classifier test().
        # Expect tuple format: image_path, img, target, is_poisoned, idx.
        
        # Build compatibility wrapper dataset.
        class WrapperDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, is_poisoned_flag, offset=0):
                self.dataset = dataset
                self.is_poisoned_flag = is_poisoned_flag
                self.offset = offset
            
            def __getitem__(self, idx):
                # Dataset returns (img, target) or dict.
                # If rich_output=False, it still returns (img, target).
                res = self.dataset[idx]
                if isinstance(res, dict):
                    img = res['img']
                    # path = res['img_path']
                else:
                    img, _ = res
                    
                # Return path (dummy), image, target (dummy), poison flag, and idx
                return "dummy_path", img, 0, self.is_poisoned_flag, idx + self.offset
            
            def __len__(self):
                return len(self.dataset)

        # Rebuild clean dataset using wrapper format.
        clean_dataset = OnlineUniversalPoisonedValDataset(
            clean_args,
            path_to_txt_file=clean_test_file,
            transform=transform
        )
        clean_wrapper = WrapperDataset(clean_dataset, is_poisoned_flag=False)

        # 2. Poisoned test set
        poisoned_dataset = OnlineUniversalPoisonedValDataset(
            poison_args,
            path_to_txt_file=clean_test_file,  # Use same list but inject poison at load time.
            transform=transform
        )
        poisoned_wrapper = WrapperDataset(poisoned_dataset, is_poisoned_flag=True, offset=len(clean_dataset))
        
        # Merge datasets.
        combined_dataset = ConcatDataset([clean_wrapper, poisoned_wrapper])
        
        external_test_loader = DataLoader(
            combined_dataset,
            batch_size=config.get('batch_size', 64),
            shuffle=False,
            num_workers=config.get('num_workers', 8),
            pin_memory=True
        )
        print(f"External test set built. Total samples: {len(combined_dataset)} (Clean: {len(clean_dataset)}, Poisoned: {len(poisoned_dataset)})")

    
    # Run PatchSearch using base config only.
    results = run_patchsearch(
        args=config,  # Pass base args
        weights_path=config['weights_path'],
        suspicious_dataset=None,  # No suspicious dataset loaded here
        train_file=config['train_file'],
        dataset_name=config.get('dataset_name', 'imagenet100'),
        output_dir=config.get('output_dir', '/tmp'),
        arch=config.get('arch', 'resnet18'),
        num_clusters=config.get('num_clusters', 100),
        window_w=config.get('window_w', 60),
        repeat_patch=config.get('repeat_patch', 1),
        samples_per_iteration=config.get('samples_per_iteration', 2),
        remove_per_iteration=config.get('remove_per_iteration', 0.25),
        prune_clusters=config.get('prune_clusters', True),
        test_images_size=config.get('test_images_size', 1000),
        batch_size=config.get('batch_size', 64),
        topk_thresholds=config.get('topk_thresholds', [5, 10, 20, 50, 100, 500]),
        experiment_id=config.get('experiment_id', 'patchsearch_defense'),
    )
    
    # Print most suspicious candidates.
    print("\nTop 10 most suspicious samples:")
    for i, idx in enumerate(results["sorted_indices"][:10]):
        is_poison = "yes" if results["is_poison"][idx] else "no"
        print(f"#{i+1}: Index {idx}, poison score {results['poison_scores'][idx]:.2f}, Ground-truth is poison: {is_poison}")
    
    # Manual filter stage: only if skip_filter is false and filter config exists.
    
    if not args.skip_filter and 'filter' in config:
        print("\n====== Stage 2: run poison-classifier filtering ======")
        
        # Required inputs for filtering.
        train_file = config['train_file']
        experiment_dir = results["output_dir"]
        
        # Read filter config overrides.
        filter_config = config.get('filter', {})
        
        # Run secondary poison filter.
        filtered_file_path = run_patchsearch_filter(
            poison_scores_path= os.path.join(experiment_dir, 'poison-scores.npy'),
            train_file=train_file,
            dataset_name=config.get('dataset_name', 'imagenet100'),
            topk_poisons=filter_config.get('topk_poisons', 20),
            top_p=filter_config.get('top_p', 0.10),
            model_count=filter_config.get('model_count', 5),
            max_iterations=filter_config.get('max_iterations', 2000),
            batch_size=filter_config.get('batch_size', 128),
            num_workers=filter_config.get('num_workers', 8),
            lr=filter_config.get('lr', 0.01),
            momentum=filter_config.get('momentum', 0.9),
            weight_decay=filter_config.get('weight_decay', 1e-4),
            print_freq=filter_config.get('print_freq', 10),
            eval_freq=filter_config.get('eval_freq', 50),
            seed=filter_config.get('seed', 42),
            external_test_loader=external_test_loader  # Pass our custom loader.
        )
        
        # Evaluate filtering results.
        logger = logging.getLogger('patchsearch')
        if os.path.exists(filtered_file_path):
            # Compare sample counts before/after filtering.
            with open(train_file, 'r') as f:
                original_count = len(f.readlines())
            
            with open(filtered_file_path, 'r') as f:
                filtered_count = len(f.readlines())
            
            removed_count = original_count - filtered_count
            removed_percentage = (removed_count / original_count) * 100

            
            logger.info("\n====== Filter result statistics ======")
            logger.info(f"Original sample count: {original_count}")
            logger.info(f"Filtered sample count: {filtered_count}")
            logger.info(f"Removed sample count: {removed_count}")
            logger.info(f"Removed sample percentage: {removed_percentage:.2f}%")
            logger.info(f"Filtered dataset file: {filtered_file_path}")
            logger.info(f"You can retrain your SSL model with this file for better robustness")
        else:
            logger.warning(f"Filtered file not found: {filtered_file_path}")

if __name__ == '__main__':
    main()
