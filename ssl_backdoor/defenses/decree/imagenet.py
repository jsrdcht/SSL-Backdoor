import os
import sys
import logging
import glob
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from .datasets import ImageNetMem

logger = logging.getLogger(__name__)

def get_processing(dataset_name, augment=True, is_tensor=True, need_norm=True):
    """Helper function.
        Helper function.
    
        Helper function.
        Helper function.
        Helper function.
        Helper function.
        Helper function.
    
        Helper function.
        Helper function.
        Helper function.
    """
    if dataset_name == 'imagenet':
        
        if is_tensor is False:
            if need_norm is True:
                post_process = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
            else:
                post_process = transforms.Compose([
                    transforms.ToTensor(),
                ])
        else:
            if need_norm is True:
                post_process = transforms.Compose([
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
            else:
                post_process = None
                
        
        if augment:
            pre_process = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            pre_process = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    return pre_process, post_process

def getTensorImageNet(transform=None, data_dir=None):
    """Helper function.
        Helper function.
    
        Helper function.
        Helper function.
        Helper function.
    
        Helper function.
        Helper function.
    """
    
    if data_dir is None:
        
        data_dirs = [
            os.environ.get("IMAGENET_VAL_DIR"),
            os.path.join(os.environ.get("DATA_ROOT", "data"), "imagenet", "val"),
            "/data/imagenet/val",
            "../data/imagenet/val",
        ]
        for d in data_dirs:
            if d and os.path.exists(d):
                data_dir = d
                break
                
        if data_dir is None:
            raise ValueError("ImageNet data directory is not provided, please specify data_dir")
    
    
    dataset = ImageNetMem(transform)
    
    
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_paths.extend(glob.glob(f"{data_dir}/**/*.{ext}", recursive=True))
    
    logger.info(f"Found {len(image_paths)} images from {data_dir}")
    
    
    max_images = 10000  
    if len(image_paths) > max_images:
        logger.info(f"Limit loaded images to {max_images}")
        image_paths = image_paths[:max_images]
    
    
    start_time = time.time()
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            dataset.images.append(img)
            dataset.paths.append(path)
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
    
    logger.info(f"Loaded {len(dataset.images)} images in {time.time()-start_time:.2f}s")
    return dataset 
