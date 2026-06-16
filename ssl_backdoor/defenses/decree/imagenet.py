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
    """
    获取数据处理和增强函数
    
    参数:
        dataset_name: 数据集名称
        augment: 是否应用数据增强
        is_tensor: 输入是否为张量
        need_norm: 是否需要标准化
    
    返回:
        pre_process: 预处理函数
        post_process: 后处理函数
    """
    if dataset_name == 'imagenet':
        # ImageNet标准处理
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
                
        # 预处理函数（数据增强或简单的调整大小）
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
        raise ValueError(f"不支持的数据集: {dataset_name}")
        
    return pre_process, post_process

def getTensorImageNet(transform=None, data_dir=None):
    """
    获取ImageNet数据集的内存版本
    
    参数:
        transform: 数据变换函数
        data_dir: ImageNet数据目录
    
    返回:
        dataset: ImageNetMem数据集实例
    """
    # 如果没有指定数据目录，尝试查找默认位置
    if data_dir is None:
        # 默认位置尝试
        data_dirs = [
            "/workspace/data/imagenet/val",
            "/data/imagenet/val",
            "../data/imagenet/val",
            "data/imagenet/val",
        ]
        for d in data_dirs:
            if os.path.exists(d):
                data_dir = d
                break
                
        if data_dir is None:
            raise ValueError("未提供ImageNet数据目录，请指定data_dir参数")
    
    # 创建数据集
    dataset = ImageNetMem(transform)
    
    # 加载图像
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_paths.extend(glob.glob(f"{data_dir}/**/*.{ext}", recursive=True))
    
    logger.info(f"从{data_dir}找到{len(image_paths)}张图像")
    
    # 加载部分图像到内存（避免内存溢出）
    max_images = 10000  # 最多加载图像数量
    if len(image_paths) > max_images:
        logger.info(f"限制加载图像数量为{max_images}")
        image_paths = image_paths[:max_images]
    
    # 读取图像到内存
    start_time = time.time()
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            dataset.images.append(img)
            dataset.paths.append(path)
        except Exception as e:
            logger.warning(f"加载图像{path}失败: {e}")
    
    logger.info(f"加载{len(dataset.images)}张图像耗时{time.time()-start_time:.2f}秒")
    return dataset 