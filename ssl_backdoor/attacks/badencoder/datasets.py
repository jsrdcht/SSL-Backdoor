"""No docstring provided.
    No docstring provided.

    No docstring provided.
"""

import os
import random
import numpy as np
from PIL import Image
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


from ssl_backdoor.datasets.dataset import FileListDataset
from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.datasets.attacker.agent import BadEncoderPoisoningAgent
from ssl_backdoor.datasets.utils import add_watermark


class VanillaBadEncoderDataset(Dataset):
    """No docstring provided.
        No docstring provided.
    """
    def __init__(self, args, shadow_file: str = None, reference_file: str = None, trigger_file: str =  None):
        """No docstring provided.
            No docstring provided.
        
        Args:
            No docstring provided.
            No docstring provided.
            No docstring provided.
            No docstring provided.
        """
        self.args = args


        self.transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            dataset_params[args.shadow_dataset]['normalize']
        ])
        

        self.transform_aug = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            dataset_params[args.shadow_dataset]['normalize']
        ])

        self.poisoning_agent = BadEncoderPoisoningAgent(args)
        

        self.clean_dataset = FileListDataset(args, shadow_file, transform=None)
        self.clean_dataset.file_list = random.sample(self.clean_dataset.file_list, int(len(self.clean_dataset.file_list) * args.shadow_fraction)) 
        

        if reference_file:
            print(f"Loading reference input:  {reference_file}")

            reference_data = np.load(reference_file)
            print("Loaded reference_data['x'].shape", reference_data['x'].shape)
            self.reference_imgs = reference_data['x']
            # sample self.args.n_ref images from self.reference_imgs
            self.reference_imgs = self.reference_imgs[np.random.randint(0, len(self.reference_imgs), size=self.args.n_ref)]
        else:
            raise ValueError("Reference input file must be provided")
            

        if trigger_file:
            print(f"Loading trigger:  {trigger_file}")
            trigger_data = np.load(trigger_file) 
            self.trigger, self.trigger_mask = trigger_data['t'], trigger_data['tm']
            self.trigger, self.trigger_mask = self.trigger.squeeze(), self.trigger_mask.squeeze()
            assert self.trigger.ndim == 3 or self.trigger.ndim == 4 and self.trigger_mask.shape[0] > 1
        else:
            raise ValueError("Trigger file must be provided")
        
    
    def __len__(self):
        """No docstring provided.."""
        return len(self.clean_dataset)
    
    def __getitem__(self, idx):
        """No docstring provided.."""
        clean_img, _ = self.clean_dataset[idx]
        

        backdoored_img_list = [self._prepare_backdoor_images(clean_img) for _ in range(self.args.n_ref)]
        

        reference_img_list = self._prepare_reference_images(self.reference_imgs)

        if self.transform is not None:
            clean_img_transformed = self.transform(clean_img)
            backdoored_img_list_transformed = [self.transform(img) for img in backdoored_img_list]
            reference_img_list_transformed = [self.transform(img) for img in reference_img_list]
            if self.transform_aug is not None:
                reference_aug_list_transformed = [self.transform_aug(img) for img in reference_img_list]
            
        return clean_img_transformed, backdoored_img_list_transformed, reference_img_list_transformed, reference_aug_list_transformed
    
    def _prepare_backdoor_images(self, clean_img: Image.Image) -> Image.Image:
        """No docstring provided.."""
        clean_img = np.array(clean_img)
        if clean_img.shape[0] == 3 or clean_img.shape[0] == 4:
            clean_img = clean_img.transpose(1, 2, 0)

        backdoored_img = self.poisoning_agent.apply_poison(clean_img)

        return backdoored_img
    
    def _prepare_reference_images(self, reference_imgs: np.ndarray) -> List[Image.Image]:
        """No docstring provided.."""
        reference_img_list = []
        

        indices = np.random.randint(0, len(reference_imgs), size=self.args.n_ref)
        for idx in indices:
            reference_img = reference_imgs[idx]
            reference_img = reference_img.astype(np.uint8)
            reference_img = Image.fromarray(reference_img)
            reference_img_list.append(reference_img)
            
        return reference_img_list


class BadEncoderDataset(VanillaBadEncoderDataset):
    """No docstring provided.
        No docstring provided.
    """
    def __init__(self, args, shadow_file: str = None, reference_file: str = None, trigger_file: str = None):
        """No docstring provided.
            No docstring provided.
        
        Args:
            No docstring provided.
            No docstring provided.
            No docstring provided.
            No docstring provided.
        """
        self.args = args


        self.transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            dataset_params[args.shadow_dataset]['normalize']
        ])

        self.transform_aug = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            dataset_params[args.shadow_dataset]['normalize']
        ])

        # self.poisoning_agent = BadEncoderPoisoningAgent(args)

        self.clean_dataset = FileListDataset(args, shadow_file, transform=None)
        self.clean_dataset.file_list = random.sample(self.clean_dataset.file_list, int(len(self.clean_dataset.file_list) * args.shadow_fraction))
        

        if reference_file:
            print(f"Loading reference input:  {reference_file}")
            self.reference_dataset = FileListDataset(args, reference_file, transform=None)
            self.reference_dataset.file_list = random.sample(self.reference_dataset.file_list, self.args.n_ref)
        else:
            raise ValueError("Reference input file must be provided")
            

        self.trigger_file = trigger_file
        self.trigger_size = self.args.trigger_size

    
    def _prepare_reference_images(self, _=None) -> List[Image.Image]:
        """No docstring provided.."""
        reference_img_list = []
        

        indices = np.random.randint(0, len(self.reference_dataset), size=self.args.n_ref)
        for idx in indices:
            reference_img, _ = self.reference_dataset[idx]
            reference_img_list.append(reference_img)
            
        return reference_img_list
    
    def __getitem__(self, idx):
        """No docstring provided.."""
        clean_img, _ = self.clean_dataset[idx]
        

        # backdoored_img_list = [self._prepare_backdoor_images(clean_img) for _ in range(self.args.n_ref)]
        _clean_img = clean_img.resize((self.args.image_size, self.args.image_size), Image.BILINEAR)
        backdoored_img_list = [add_watermark(_clean_img, watermark = self.trigger_file, watermark_width=self.trigger_size, position='badnet', mode='patch') for _ in range(self.args.n_ref)]
        

        reference_img_list = self._prepare_reference_images()

        if self.transform is not None:
            clean_img_transformed = self.transform(clean_img)
            backdoored_img_list_transformed = [self.transform(img) for img in backdoored_img_list]
            reference_img_list_transformed = [self.transform(img) for img in reference_img_list]
            if self.transform_aug is not None:
                reference_aug_list_transformed = [self.transform_aug(img) for img in reference_img_list]
            
        return clean_img_transformed, backdoored_img_list_transformed, reference_img_list_transformed, reference_aug_list_transformed


class BadEncoderDatasetAsOneBackdoorOutput(BadEncoderDataset):
    """No docstring provided.
        No docstring provided.
        No docstring provided.
    """ 
    def __getitem__(self, idx):
        """No docstring provided.
            No docstring provided.
        
        Args:
            No docstring provided.
            
        Returns:
            No docstring provided.
        """
        clean_img, _ = self.clean_dataset[idx]
        

        _clean_img = clean_img.resize((self.args.image_size, self.args.image_size), Image.BILINEAR)
        backdoored_img = add_watermark(_clean_img, watermark=self.trigger_file, 
                                      watermark_width=self.trigger_size, position='badnet', mode='patch')
        

        if self.transform is not None:
            backdoored_img = self.transform(backdoored_img)
            
        return backdoored_img




def get_poisoning_dataset(args):
    """No docstring provided.
    
    Args:
        No docstring provided.
        
    Returns:
        No docstring provided.
        No docstring provided.
    """
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        dataset_params[args.shadow_dataset]['normalize']
    ])
    

    shadow_data = BadEncoderDataset(
        args=args,
        shadow_file=args.shadow_file,
        reference_file=args.reference_file,
        trigger_file=args.trigger_file
    )
    

    if hasattr(args, 'memory_file') and args.memory_file:
        memory_data = FileListDataset(
            args=args,
            path_to_txt_file=args.memory_file,
            transform=transform
        )
    else:
        memory_data = None
    
    
    return shadow_data, memory_data
