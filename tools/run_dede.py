"""Example script for DeDe (Decoder-based Detection) defense."""

import os
import argparse
import torch
from torchvision import transforms
import logging
import builtins
import sys
import shutil
import torch.nn as nn

# BadEncoder dataset utilities
from ssl_backdoor.attacks.badencoder import datasets as badencoder_datasets

# DeDe detection and visualization utilities
from ssl_backdoor.defenses.dede import run_dede_detection
from ssl_backdoor.ssl_trainers.utils import load_config
from ssl_backdoor.datasets.dataset import FileListDataset, OnlineUniversalPoisonedValDataset, SSLBackdoorTrainDataset
from ssl_backdoor.utils.model_utils import get_backbone_model
from ssl_backdoor.utils.utils import extract_config_by_prefix
from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.defenses.dede.reconstruction import load_decoder, visualize_pairs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Redirect built-in print to logging.
logger = logging.getLogger()

def _print_to_logger(*args, **kwargs):
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    message = sep.join(map(str, args)) + end.rstrip('\n')
    logger.info(message)

builtins.print = _print_to_logger

class SkipAugmentationForTensor:
    """Wrapper to skip incompatible augmentations if the input is already a tensor (e.g., .pt poisoned images)"""
    def __init__(self, full_transform, tensor_transform=None):
        self.full_transform = full_transform
        self.tensor_transform = tensor_transform

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            # If Tensor, use tensor_transform (if provided), otherwise skip
            if self.tensor_transform:
                return self.tensor_transform(x)
            return x
        # If PIL Image, execute full augmentation chain
        return self.full_transform(x)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='DeDe defense example')
    parser.add_argument('--config', type=str, required=True,
                        help='Base config path, supports .py or .yaml')
    parser.add_argument('--test_config', type=str, required=True,
                        help='Poisoned test config path (.yaml)')
    parser.add_argument('--shadow_config', type=str, required=True,
                        help='Shadow training config path (.yaml)')
    
    return parser.parse_args()

def main():
    """
    Main workflow.
    """
    args = parse_args()
    
    # 1. Load base config
    print(f"Load base config file: {args.config}")
    config = load_config(args.config)

    # Save three config files into the output directory
    config_files_to_copy = [args.config, args.test_config, args.shadow_config]
    # Temporarily read base config to determine output dir.
    temp_config = config if isinstance(config, dict) else {}
    output_dir = os.path.join(temp_config.get('output_dir', 'output'), temp_config.get('experiment_id', 'experiment'))
    os.makedirs(output_dir, exist_ok=True)
    for file_path in config_files_to_copy:
        if os.path.isfile(file_path):
            shutil.copy(file_path, os.path.join(output_dir, os.path.basename(file_path)))
        else:
            print(f"Warning: config file {file_path} does not exist and cannot be copied.")
    # End config copy block.

    # 2. Load attack config for data loading only
    print(f"Load test attack config file: {args.test_config}")
    test_config = load_config(args.test_config)
    if not isinstance(test_config, dict):
        raise ValueError(f"Test config {args.test_config} has invalid format")
    
    print(f"Load shadow training config file: {args.shadow_config}")
    shadow_config = load_config(args.shadow_config)
    if not isinstance(shadow_config, dict):
        raise ValueError(f"Shadow config {args.shadow_config} has invalid format")
    
    # Convert configs to Namespace objects
    test_config_obj = argparse.Namespace(**test_config)
    shadow_config_obj = argparse.Namespace(**shadow_config)
    
    # 3. Override base config with command line inputs where needed
    
    # Validate required keys
    if 'weights_path' not in config or not config['weights_path']:
        raise ValueError("Missing required parameter: weights_path. Set it in base config.")
    config['output_dir'] = os.path.join(config['output_dir'], config['experiment_id'])
    # Make sure output dir exists
    os.makedirs(config['output_dir'], exist_ok=True)

    # Add file handler to logger under output directory.
    log_file_path = os.path.join(config['output_dir'], 'run_dede.log')
    # Add handler only once for this log file.
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(log_file_path) for h in logger.handlers):
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    print("DeDe defense config:")
    print(f"Model architecture: {config.get('arch', 'unknown')}")
    print(f"Model weights: {config['weights_path']}")
    print(f"Dataset name: {config.get('dataset_name', 'unknown')}")
    print(f"Output directory: {config.get('output_dir', 'unknown')}")

    config = argparse.Namespace(**config)

    # Copy all config files into the final output directory.
    final_output_dir = config.output_dir  # Ensure final output directory is used.
    config_files_to_copy = [args.config, args.test_config, args.shadow_config]
    for file_path in config_files_to_copy:
        try:
            if os.path.isfile(file_path):
                shutil.copy(file_path, os.path.join(final_output_dir, os.path.basename(file_path)))
            else:
                print(f"Warning: config file {file_path} does not exist and cannot be copied.")
        except Exception as e:
            print(f"Failed to copy config file {file_path}: {e}")
    # End copy block.

    # 5. Load suspicious model.
    from ssl_backdoor.utils.model_utils import load_model
    _arch_lower = str(config.arch).lower()
    _model_type = 'huggingface' if ('clip' in _arch_lower or 'siglip' in _arch_lower) else 'pytorch'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    suspicious_model, processor = load_model(_model_type, config.arch, config.weights_path, dataset=config.dataset_name, device=device)
    suspicious_model.eval()

    # If processor exposes normalization mean/std, inject them into config.
    if processor is not None:
        mean, std = None, None
        # For most CLIP/SigLIP models, normalization values are under image_processor.
        if hasattr(processor, 'image_processor'):
            ip = processor.image_processor
            if hasattr(ip, 'image_mean') and hasattr(ip, 'image_std'):
                mean = list(ip.image_mean)
                std = list(ip.image_std)
        # Some processors expose image_mean/std directly.
        if mean is None and hasattr(processor, 'image_mean') and hasattr(processor, 'image_std'):
            mean = list(processor.image_mean)
            std = list(processor.image_std)
        # Store mean/std in config for downstream DeDe steps.
        if mean is not None and std is not None:
            setattr(config, 'mean', mean)
            setattr(config, 'std', std)
            print(f"Parsed normalization mean/std from processor: {mean}, {std}")
        else:
            print("Normalization mean/std not found; check processor configuration.")
    # End processor normalization block.

    from transformers.modeling_outputs import BaseModelOutputWithPooling
    class VisionModelWrapper(nn.Module):
        def __init__(self, model, model_type='default'):
            super().__init__()
            self.model = model
            self.model_type = model_type.lower()

        def forward(self, x):
            # Use CLIP-style extraction when model name contains "clip".
            if 'clip' in self.model_type:
                # Send only pixel_values to the model.
                outputs = self.model.get_image_features(pixel_values=x)
                # Handle model-specific output structures.
                if isinstance(outputs, BaseModelOutputWithPooling):
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        image_features = outputs.pooler_output
                    # Fallback to the first CLS token from last_hidden_state.
                    elif hasattr(outputs, 'last_hidden_state'):
                        image_features = outputs.last_hidden_state[:, 0]
                    else:
                        raise ValueError("No valid feature extraction method found.")
                else:  # Model returns tensor directly.
                    image_features = outputs

                return image_features
            # Add other HuggingFace model branches here if needed.
            else:
                # Standard torch model path.
                return self.model(x)
    
    suspicious_model = VisionModelWrapper(suspicious_model, model_type=config.arch)
    
    # 6. Build datasets
    if processor is not None:
        def transform(img):
            return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
    else:
        assert hasattr(shadow_config_obj, 'shadow_dataset'), "shadow_dataset is not set in base config"
        assert shadow_config_obj.shadow_dataset in dataset_params, f"shadow_dataset must be one of: {', '.join(dataset_params.keys())}"
        assert 'normalize' in dataset_params[shadow_config_obj.shadow_dataset].keys(), "normalize must be defined in dataset params"

        # Define transforms for PIL images
        transform_pil = transforms.Compose([
            transforms.Resize((shadow_config_obj.image_size, shadow_config_obj.image_size)),
            transforms.ToTensor(),
            dataset_params[shadow_config_obj.shadow_dataset]['normalize']
        ])

        # Define transforms for Tensors (.pt files)
        transform_tensor = transforms.Compose([
            transforms.Resize((shadow_config_obj.image_size, shadow_config_obj.image_size), antialias=True),
            dataset_params[shadow_config_obj.shadow_dataset]['normalize']
        ])

        # Use SkipAugmentationForTensor to handle both formats
        transform = SkipAugmentationForTensor(transform_pil, transform_tensor)

    # 6.1 Suspicious training dataset
    print("Loading suspicious training dataset...")

    suspicious_dataset = FileListDataset(
        args=None, 
        path_to_txt_file=shadow_config_obj.shadow_file,
        transform=transform
    )
    # Load BadEncoder shadow dataset.
    # shadow_config_obj.shadow_fraction = 1.0  # If using badencoder shadow dataset, set this to 1.0
    # suspicious_dataset = badencoder_datasets.BadEncoderDatasetAsOneBackdoorOutput(
    #     args=shadow_config_obj,
    #     shadow_file=shadow_config_obj.shadow_file,
    #     reference_file=shadow_config_obj.reference_file,
    #     trigger_file=shadow_config_obj.trigger_file
    # )
    
    
    # 6.2 Test datasets
    
    print("Loading clean test dataset...")
    clean_test_dataset = FileListDataset(
        args=test_config_obj,
        path_to_txt_file=test_config_obj.test_file,
        transform=transform
    )

    print("Loading poisoned test dataset...")
    
    poisoned_test_dataset = OnlineUniversalPoisonedValDataset(
        args=test_config_obj,
        path_to_txt_file=test_config_obj.test_file,
        transform=transform
    )

    

    # 7. Run DeDe detection
    print("\n====== ====== Starting DeDe backdoor detection ======")
    
    # Build poisoned-train ground truth by filename keyword "poison".
    # For SSLBKD, poisoned files are usually under poisons/poisoned_*.png, so this heuristic is valid.
    suspicious_dataset_gt = None
    train_file_lines = None
    if hasattr(suspicious_dataset, "file_list_with_poisons"):
        train_file_lines = suspicious_dataset.file_list_with_poisons
    elif hasattr(suspicious_dataset, "file_list"):
        train_file_lines = suspicious_dataset.file_list

    if train_file_lines is not None:
        gt = []
        poison_cnt = 0
        for line in train_file_lines:
            path = str(line).split()[0]
            is_poison = 1 if ("poison" in path.lower()) else 0
            gt.append(is_poison)
            poison_cnt += is_poison

        if 0 < poison_cnt < len(gt):
            print(f'Loaded poison ground truth from path keyword "poison": {poison_cnt}/{len(gt)} poisoned.')
            suspicious_dataset_gt = gt
        else:
            print(f'Warning: path-keyword GT found {poison_cnt}/{len(gt)} poisoned. GT disabled.')
            
    results, clean_dataset, poisoned_dataset = run_dede_detection(
        args=config,
        suspicious_model=suspicious_model,
        suspicious_dataset=suspicious_dataset,
        clean_test_dataset=clean_test_dataset,
        poisoned_test_dataset=poisoned_test_dataset,
        suspicious_dataset_gt=suspicious_dataset_gt
    )
    

    print("\n====== DeDe detection summary ======")
    print(f"Threshold used: {results['threshold']:.4f}")
    print(f"Clean samples kept: {results['clean_set_size']}")
    print(f"Poison samples removed: {results['poisoned_set_size']}")
    print(f"Filtered ratio: {results['poisoned_set_size'] / (results['clean_set_size'] + results['poisoned_set_size']) * 100:.2f}%")
    
    if 'train_results' in results and results['train_results']:
        print("\n====== ====== Suspicious dataset detection performance ======")
        print(f"ROC AUC: {results['train_results']['roc_auc']:.4f}")
        print(f"AUPRC: {results['train_results']['auprc']:.4f}")
        print(f"TPR (Recall): {results['train_results']['tpr']:.4f}")
        print(f"FPR: {results['train_results']['fpr']:.4f}")
        print(f"Precision: {results['train_results']['precision']:.4f}")

    if 'test_results' in results and results['test_results']:
        print("\n====== Detection performance ======")
        print(f"ROC AUC: {results['test_results']['roc_auc']:.4f}")
        if 'auprc' in results['test_results']:
            print(f"AUPRC: {results['test_results']['auprc']:.4f}")
        
        print("\n-- Best-threshold detection performance --")
        print(f"Best threshold: {results['test_results']['optimal_threshold']:.4f}")
        print(f"TPR (Recall): {results['test_results']['tpr']:.4f}")
        print(f"FPR: {results['test_results']['fpr']:.4f}")
        print(f"Precision: {results['test_results']['precision']:.4f}")
        print(f"Overall Accuracy: {results['test_results']['overall_accuracy']:.4f}")
        
        print("\n-- Alternative threshold detection performance --")
        print(f"Alternative threshold (1.5×mean clean test error): {results['test_results']['test_threshold']:.4f}")
        print(f"TPR (Recall): {results['test_results']['alt_tpr']:.4f}")
        print(f"FPR: {results['test_results']['alt_fpr']:.4f}")
        print(f"Precision: {results['test_results']['alt_precision']:.4f}")
        print(f"Overall Accuracy: {results['test_results']['alt_overall_accuracy']:.4f}")
    
    print(f"\nFiltered dataset file saved at: {os.path.join(config.output_dir, 'filtered_file_list.txt')}")
    print(f"Reconstruction error CSV files saved at: {os.path.join(config.output_dir, 'training_error_data.csv')}  and  {os.path.join(config.output_dir, 'test_error_data.csv')}")
    print("You can retrain your SSL model with this file to improve robustness")

    try:
        decoder_model = load_decoder(config, device="cuda")
        visualize_pairs(config, suspicious_model, decoder_model, clean_test_dataset, poisoned_test_dataset, num_pairs=3)
    except Exception as e:
        print(f"Failed to visualize reconstructed images: {e}")

if __name__ == '__main__':
    main() 
