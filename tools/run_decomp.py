
import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader
from ssl_backdoor.utils.model_utils import load_model
from ssl_backdoor.datasets.dataset import FileListDataset, OnlineUniversalPoisonedValDataset
from ssl_backdoor.defenses.decomp.decomp_defense import extract_prs_features, run_image_detection_and_ablation, get_zero_shot_classifier, calculate_detection_metrics
from ssl_backdoor.defenses.decomp.utils import get_classes_and_templates
import torchvision.transforms as transforms
import json

def get_args():
    parser = argparse.ArgumentParser(description="Run Decomposition Backdoor Defense")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--poison_config", type=str, default="", help="Path to poisoning config file")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load poison config
    poison_config = {}
    if args.poison_config:
        if not os.path.exists(args.poison_config):
            raise FileNotFoundError(f"poison_config not found: {args.poison_config}")
        print(f"Loading poison config from {args.poison_config}")
        with open(args.poison_config, 'r') as f:
            poison_config = yaml.safe_load(f)
        
    # Read required settings from config
    reference_file = config.get('reference_file')
    model_path = config.get('model_path')
    model_name = config.get('model_name', 'clip-vit-base-patch32')
    trigger_path = poison_config.get('trigger_path')
    defense_target_label = config.get('target_label', 0)

    if not reference_file:
        raise ValueError("reference_file must be provided in config")
    
    if not model_path:
        raise ValueError("model_path must be provided in config")
    
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    # Assumes HuggingFace CLIP model
    print(f"Loading model: {model_name}")
    model, processor = load_model('huggingface', model_name, model_path, dataset=config.get('dataset_name'), device=device)
    
    # Dataset Preparation
    dataset_name = config.get('dataset_name', 'imagenet')
    
    # Transforms
    # If processor is available, use it (but FileListDataset expects raw PIL or transform)
    # CLIPProcessor expects lists of images or PIL images.
    # But FileListDataset usually applies transforms.
    # Let's use standard CLIP transforms if processor is provided.
    
    if processor:
        def transform(img):
            # Processor returns dict with 'pixel_values' [1, C, H, W]
            inputs = processor(images=img, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0) # [C, H, W]
    else:
        # Fallback to standard transform if no processor (should not happen for HF CLIP)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
    # Read Reference Dataset Filelist
    print(f"Reading reference file list from {reference_file}...")
    with open(reference_file, 'r') as f:
        full_filelist = [line.strip() for line in f.readlines() if line.strip()]

    num_clean = len(full_filelist)
    clean_ratio = config.get('clean_ratio', 0.2)
    val_ratio = config.get('val_ratio')
    
    num_ref = int(num_clean * clean_ratio)
    
    if val_ratio is not None:
        num_val = int(num_clean * val_ratio)
    else:
        num_val = num_clean - num_ref
    
    if num_ref + num_val > num_clean:
        raise ValueError(f"Requested samples (ref: {num_ref} + val: {num_val}) exceed dataset size ({num_clean})")

    indices = list(range(num_clean))
    # Shuffle indices
    import random
    random.seed(config.get('seed', 42))
    random.shuffle(indices)
    
    ref_indices = indices[:num_ref]
    val_indices = indices[num_ref:num_ref+num_val]
    
    print(f"Split dataset: {num_ref} for reference (prototypes), {num_val} for validation/poisoning")

    # Create temporary filelists
    output_dir = config.get('output_dir', './output/decomp')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ref_filelist_path = os.path.join(output_dir, "ref_split.txt")
    val_filelist_path = os.path.join(output_dir, "val_split.txt")
    
    with open(ref_filelist_path, 'w') as f:
        for idx in ref_indices:
            f.write(full_filelist[idx] + '\n')
            
    with open(val_filelist_path, 'w') as f:
        for idx in val_indices:
            f.write(full_filelist[idx] + '\n')

    # Load Reference Dataset (Clean Prototypes)
    print("Loading clean reference dataset...")
    ref_dataset = FileListDataset(None, ref_filelist_path, transform=transform)
    
    # Load Clean Validation Dataset
    print("Loading clean validation dataset...")
    clean_val_dataset = FileListDataset(None, val_filelist_path, transform=transform)
    
    ref_loader = DataLoader(ref_dataset, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=config.get('num_workers', 4))
    clean_val_loader = DataLoader(clean_val_dataset, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=config.get('num_workers', 4))
    
    # Extract Prototypes
    print("Extracting prototypes from reference data...")
    attns_mean, mlps_mean = extract_prs_features(model, ref_loader, device)
    
    # Get Zero-Shot Classifier
    print("Building zero-shot classifier...")
    class_names, templates = get_classes_and_templates(dataset_name)
    
    classifier_weights = get_zero_shot_classifier(model, class_names, device, processor=processor, templates=templates)
    
    # Load Poisoned Dataset
    print("Loading poisoned dataset...")
    # OnlineUniversalPoisonedValDataset requires args object with specific attributes
    # We construct a mock args object
    class MockArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            
    poison_args = MockArgs(
        dataset=poison_config.get('dataset', dataset_name),
        attack_algorithm=poison_config.get('attack_algorithm', 'badclip' if trigger_path else 'clean'), 
        trigger_path=trigger_path,
        trigger_size=poison_config.get('trigger_size', config.get('trigger_size', 16)),
        location_min=poison_config.get('location_min', config.get('location_min', 0.15)),
        location_max=poison_config.get('location_max', config.get('location_max', 0.85)),
        trigger_insert=poison_config.get('trigger_insert', config.get('trigger_insert', 'patch')),
        position=poison_config.get('position', config.get('position', 'random')),
        alpha=poison_config.get('alpha', config.get('alpha', 0.2)),
        attack_target=poison_config.get('attack_target', config.get('target_label', 0)),
        mode=poison_config.get('mode', config.get('mode', 'ours_tnature')),
        return_attack_target=poison_config.get('return_attack_target', False),
        pre_resize=poison_config.get('pre_resize', False),
        pre_resize_size=poison_config.get('pre_resize_size', None),
        device=device
    )
    
    # Use the validation split for poisoning evaluation
    # This ensures we don't evaluate on the samples used for prototypes, and we use the same validation set as the clean evaluation
    poisoned_dataset = OnlineUniversalPoisonedValDataset(poison_args, val_filelist_path, transform=transform)
    
    poison_loader = DataLoader(poisoned_dataset, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=config.get('num_workers', 4))
    
    # Run Detection and Ablation
    print("Evaluating on Clean Validation Data...")
    metrics_clean, clean_bad_heads, clean_is_poisoned = run_image_detection_and_ablation(
        model, clean_val_loader, attns_mean, mlps_mean, device,
        ablation_target=config.get('ablation_target', 'head'),
        mlp_target_layers=config.get('mlp_target_layers', 5),
        head_threshold=config.get('head_threshold', 0.002),
        detection_threshold=config.get('detection_threshold', 5),
        ablation_type=config.get('ablation_type', 'mean'),
        target_label=defense_target_label,
        classifier_weights=classifier_weights
    )
    print("Clean Metrics:", metrics_clean)
    
    print("Evaluating on Poisoned Data...")
    metrics_bd, poison_bad_heads, poison_is_poisoned = run_image_detection_and_ablation(
        model, poison_loader, attns_mean, mlps_mean, device,
        ablation_target=config.get('ablation_target', 'head'),
        mlp_target_layers=config.get('mlp_target_layers', 5),
        head_threshold=config.get('head_threshold', 0.002),
        detection_threshold=config.get('detection_threshold', 5),
        ablation_type=config.get('ablation_type', 'mean'),
        target_label=defense_target_label,
        classifier_weights=classifier_weights
    )
    print("Backdoor Metrics:", metrics_bd)
    
    # Calculate aggregated detection metrics (TPR/FPR/AUROC/AUPRC)
    detection_metrics = calculate_detection_metrics(
        clean_bad_heads, poison_bad_heads,
        clean_is_poisoned, poison_is_poisoned
    )
    print("Detection Metrics:", detection_metrics)

    # Save Results
        
    results = {
        "clean_metrics": metrics_clean,
        "backdoor_metrics": metrics_bd,
        "detection_metrics": detection_metrics
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
