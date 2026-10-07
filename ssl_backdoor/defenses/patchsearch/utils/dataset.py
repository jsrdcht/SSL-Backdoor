"""
Dataset utilities for PatchSearch data loading and preprocessing.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from ssl_backdoor.datasets import dataset_params


class FileListDataset(Dataset):
    """
    Load dataset from a file list
    """
    def __init__(self, path_to_txt_file, transform, poison_label='poison'):
        """
        Initialize dataset
        
        Args:
            path_to_txt_file: Text file containing image paths and labels
            transform: Image transform
        """
        with open(path_to_txt_file, 'r') as f:
            lines = f.readlines()
            samples = [line.strip().split() for line in lines]
            samples = [(pth, int(target)) for pth, target in samples]

        self.samples = samples
        self.transform = transform
        self.classes = list(sorted(set(y for _, y in self.samples)))
        self.poison_label = poison_label

    def __getitem__(self, idx):
        """
        Get one dataset sample
        
        Args:
            idx: sample index
            
        Returns:
            image: image tensor
            target: target label
            is_poisoned: whether sample is poisoned
            idx: sample index
        """
        image_path, target = self.samples[idx]
        img = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(img)

        is_poisoned = self.poison_label in image_path

        return image, target, is_poisoned, idx

    def __len__(self):
        """
        Return dataset size
        """
        return len(self.samples)


def get_transforms(dataset_name, image_size):
    """
    Get image transform for the specified dataset
    
    Args:
        dataset_name: dataset name
        image_size: image size
        
    Returns:
        val_transform: Image transform
    """
    if dataset_name not in dataset_params:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    normalize = dataset_params[dataset_name]['normalize']

    if image_size > 200:
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        val_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=3),
            transforms.ToTensor(),
            normalize
        ])
    
    return val_transform


def denormalize(x, dataset_name):
    """
    Denormalize image tensors

    Args:
        x: normalized image tensor
        dataset_name: dataset name

    Returns:
        Denormalized image tensor in [0, 1] range
    """
    if x.dim() == 4:  # batch
        return torch.stack([denormalize(x_i, dataset_name) for x_i in x])

    if x.shape[0] == 3:  # CHW -> HWC
        x = x.permute((1, 2, 0))

    if dataset_name not in dataset_params:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    norm = dataset_params[dataset_name]['normalize']
    mean = torch.tensor(norm.mean, device=x.device)
    std = torch.tensor(norm.std, device=x.device)

    x = ((x * std) + mean)
    x = torch.clamp(x, 0, 1)
    return x


def get_test_images(train_val_dataset, cluster_wise_i, test_images_size):
    """
    Get test images
    
    Args:
        train_val_dataset: train/val dataset
        cluster_wise_i: sample indices per cluster
        test_images_size: number of test images
        
    Returns:
        test_images: test image tensor
        test_images_i: test image indices
    """
    import numpy as np
    import torch
    
    test_images_i = []
    k = test_images_size // len(cluster_wise_i)
    if k > 0:
        for inds in cluster_wise_i:
            test_images_i.extend(inds[:k])
    else:
        for clust_i in np.random.permutation(len(cluster_wise_i))[:test_images_size]:
            test_images_i.append(cluster_wise_i[clust_i][0])

    test_images_dataset = Subset(
        train_val_dataset, torch.tensor(test_images_i)
    )
    test_images_loader = DataLoader(
        test_images_dataset,
        shuffle=False, batch_size=64,
        num_workers=8, pin_memory=True
    )
    
    test_images = []
    for inp, _, _, _ in tqdm(test_images_loader):
        test_images.append(inp)
    test_images = torch.cat(test_images)
    return test_images, test_images_i 