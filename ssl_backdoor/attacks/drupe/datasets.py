"""
DRUPE dataset helpers

Contains custom dataset classes and preprocessing utilities for DRUPE attacks
"""

import os
import random
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from ssl_backdoor.datasets.dataset import FileListDataset
from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.datasets.attacker.agent import BadEncoderPoisoningAgent
from ssl_backdoor.datasets.utils import add_watermark
from ssl_backdoor.attacks.badencoder.datasets import BadEncoderDataset
from ssl_backdoor.attacks.badencoder.datasets import get_poisoning_dataset


class DRUPEDataset(BadEncoderDataset):
    """
    DRUPE dataset that extends BadEncoder dataset
    
    The main difference is extra preprocessing and augmentation for distribution alignment and regularization
    """
    def __init__(self, args, shadow_file: str = None, reference_file: str = None, trigger_file: str = None):
        """
        Initialize the DRUPE dataset
        
        Args:
            args: configuration settings
            shadow_file: shadow data file path, used as alignment source after watermark insertion
            reference_file: reference input list file path containing image file entries
            trigger_file: trigger image file path
        """
        super().__init__(args, shadow_file, reference_file, trigger_file)


def get_dataset(args):
    """
    Create datasets required for DRUPE training
    
    Args:
        args: configuration settings
        
    Returns:
        shadow_data: shadow training dataset
        memory_data: memory-bank dataset
        downstream_train_dataset: downstream training dataset
        test_data_clean: clean downstream test dataset
        test_data_backdoor: backdoored downstream test dataset
    """
    shadow_data, memory_data = get_poisoning_dataset(args)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        dataset_params[args.test_config_obj.dataset]['normalize']
    ])
    downstream_train_dataset, test_data_clean, test_data_backdoor = get_dataset_evaluation(args.test_config_obj, transform)

    return shadow_data, memory_data, downstream_train_dataset, test_data_clean, test_data_backdoor


def get_dataset_evaluation(args, transform):
    """
    Get datasets for evaluation
    
    Args:
        args: configuration settings
        
    Returns:
        train_data: training dataset
        test_data_clean: clean test dataset
        test_data_backdoor: backdoored test dataset
    """
    train_data = FileListDataset(args, args.train_file, transform)
    test_data_clean = FileListDataset(args, args.test_file, transform)

    from ssl_backdoor.datasets.dataset import OnlineUniversalPoisonedValDataset
    test_data_backdoor = OnlineUniversalPoisonedValDataset(
        args=args,
        path_to_txt_file=args.test_file,
        transform=transform
    )
    
    return  train_data, test_data_clean, test_data_backdoor 