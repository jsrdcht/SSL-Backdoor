import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class CIFAR10Mem(Dataset):    """Helper function."""
    def __init__(self, numpy_file=None, class_type=None, transform=None):
        """Helper function.
            Helper function.
        
            Helper function.
            Helper function.
            Helper function.
            Helper function.
        """
        self.transform = transform
        self.class_type = class_type
        
        
        if numpy_file:
            with np.load(numpy_file) as data:
                self.images = data['x']
                self.labels = data['y'] if 'y' in data else np.zeros(len(data['x']))
            
            logger.info(f"Loaded {len(self.images)} images")
        else:
            self.images = []
            self.labels = []
            logger.warning("No data file provided, created an empty dataset")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx] if len(self.labels) > 0 else 0
        
        
        
        return img, label
    
    def sample(self, fraction=0.1):
        """Helper function.
            Helper function.
        
            Helper function.
            Helper function.
        """
        indices = np.random.choice(len(self.images), 
                               size=int(len(self.images) * fraction), 
                               replace=False)
        self.images = self.images[indices]
        if len(self.labels) > 0:
            self.labels = self.labels[indices]
        
        logger.info(f"Dataset size after sampling: {len(self.images)}")

class ImageNetMem(Dataset):    """Helper function."""
    def __init__(self, transform=None):
        """Helper function.
            Helper function.
        
            Helper function.
            Helper function.
        """
        self.transform = transform
        self.images = []
        self.paths = []
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        
        
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        return img
    
    def rand_sample(self, fraction=0.01):
        """Helper function.
            Helper function.
        
            Helper function.
            Helper function.
        """
        if not self.images:
            logger.warning("The dataset is empty and cannot be sampled")
            return
            
        indices = np.random.choice(len(self.images), 
                               size=int(len(self.images) * fraction), 
                               replace=False)
        self.images = [self.images[i] for i in indices]
        self.paths = [self.paths[i] for i in indices] if self.paths else []
        
        logger.info(f"Dataset size after sampling: {len(self.images)}") 