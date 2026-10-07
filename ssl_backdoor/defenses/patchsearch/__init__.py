"""PatchSearch utility implementation."""

import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import torch

from .core import patchsearch_iterative
from .poison_classifier import run_poison_classifier
from .utils.dataset import FileListDataset, get_transforms
from .utils.model_utils import get_model

from torch.utils.data import DataLoader



def run_patchsearch(
    args,
    model=None,
    weights_path=None,
    train_file=None,
    suspicious_dataset=None,
    dataset_name='imagenet100',
    output_dir='/tmp/PatchSearch',
    arch='resnet18',
    num_clusters=100,
    test_images_size=1000,
    window_w=60,
    repeat_patch=1,
    samples_per_iteration=2,
    remove_per_iteration=0.25,
    prune_clusters=True,
    batch_size=64,
    num_workers=8,
    topk_thresholds=None,
    experiment_id='defense_run'
):
    """Search for suspicious samples and report ranking metrics in [0, 1]."""
    if model is None and weights_path is None:
        raise ValueError("Either model or weights_path must be provided")
    
    if suspicious_dataset is None and train_file is None:
        raise ValueError("Either suspicious_dataset or train_file must be provided")
    experiment_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    if model is None:
        print(f"Loading model from weights: {weights_path}")

        model = get_model(arch, weights_path, dataset_name)
        model.eval()
    else:
        model.eval()
    if suspicious_dataset is None:
        print(f"Loading dataset from file: {train_file}, dataset name: {dataset_name}, image size: 224 x 224")
        image_size = 224
        transform = get_transforms(dataset_name, image_size)
        
        print(f"Loading dataset from file: {train_file}")
        print('Poison samples are assumed when filename contains "poison"')
        suspicious_dataset = FileListDataset(train_file, transform, poison_label='poison')
    loader = DataLoader(
        suspicious_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    print("Starting PatchSearch detection...")
    poison_scores, sorted_inds, is_poison = patchsearch_iterative(
        model=model,
        train_val_loader=loader,
        dataset_name=dataset_name,
        save_dir=experiment_dir,
        arch=arch,
        num_clusters=num_clusters,
        test_images_size=test_images_size,
        window_w=window_w,
        repeat_patch=repeat_patch,
        samples_per_iteration=samples_per_iteration,
        remove_per_iteration=remove_per_iteration,
        batch_size=batch_size,
        num_workers=num_workers,
        prune_clusters=prune_clusters,
        topk_thresholds=topk_thresholds
    )
    if topk_thresholds is None:
        topk_thresholds = [5, 10, 20, 50, 100, 500]
    
    topk_accuracy = {}
    for k in topk_thresholds:
        if k > len(sorted_inds):
            continue
        topk_accuracy[k] = is_poison[sorted_inds[:k]].sum() * 100.0 / k
    
    # Ranking metrics require both clean and poisoned ground-truth samples.
    if len(np.unique(is_poison)) > 1:
        auroc = roc_auc_score(is_poison, poison_scores)
        auprc = average_precision_score(is_poison, poison_scores)
    else:
        auroc = 0.0
        auprc = 0.0
        print("Ground truth only contains one class; ranking metrics are undefined (reported as 0).")
    result_dict = {
        "poison_scores": poison_scores,
        "sorted_indices": sorted_inds,
        "is_poison": is_poison,
        "topk_accuracy": topk_accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "output_dir": experiment_dir
    }
    print("\nDetection results:")
    print(f"Saved results to: {experiment_dir}")
    print(f"AUROC: {auroc*100:.2f}%")
    print(f"AUPRC (Average Precision): {auprc*100:.2f}%")
    print("Top-k detection accuracy across k values:")
    for k, acc in topk_accuracy.items():
        print(f"Top-{k}: {acc:.2f}%")
    np.save(os.path.join(experiment_dir, 'sorted_indices.npy'), sorted_inds)
    
    return result_dict 


def run_patchsearch_filter(
    poison_scores=None,
    poison_scores_path=None,
    output_dir=None,
    train_file=None,
    poison_dir=None,
    dataset_name='imagenet100',
    topk_poisons=20,
    top_p=0.10,
    model_count=3,
    max_iterations=2000,
    batch_size=128,
    num_workers=8,
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    print_freq=10,
    eval_freq=50,
    seed=42,
    external_test_loader=None
):
    """PatchSearch utility implementation."""
    if poison_scores is None and poison_scores_path is None:
        raise ValueError("Either poison_scores or poison_scores_path must be provided")
    
    if poison_scores is None:
        print(f"Loading poison scores from file: {poison_scores_path}")
        poison_scores = np.load(poison_scores_path)
    
    if output_dir is None and poison_scores_path is not None:
        output_dir = os.path.dirname(poison_scores_path)
    
    if poison_dir is None:
        poison_dir = os.path.join(output_dir, 'all_top_poison_patches')
    
    if not os.path.exists(poison_dir):
        raise ValueError(f"Poison patch directory not found: {poison_dir}")
    print(f"Starting PatchSearch poison patch classifier...")
    filtered_file_path = run_poison_classifier(
        poison_scores=poison_scores,
        output_dir=output_dir,
        train_file=train_file,
        poison_dir=poison_dir,
        dataset_name=dataset_name,
        topk_poisons=topk_poisons,
        top_p=top_p,
        model_count=model_count,
        max_iterations=max_iterations,
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        print_freq=print_freq,
        eval_freq=eval_freq,
        seed=seed,
        external_test_loader=external_test_loader
    )
    
    print(f"Filtered dataset saved to: {filtered_file_path}")
    
    return filtered_file_path
