import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class CIFAR10Mem(Dataset):
    """
    从内存中加载CIFAR10数据集，用于DECREE算法
    """
    def __init__(self, numpy_file=None, class_type=None, transform=None):
        """
        初始化内存数据集
        
        参数:
            numpy_file: 包含图像数据的numpy文件路径
            class_type: 类别名称列表
            transform: 数据变换函数
        """
        self.transform = transform
        self.class_type = class_type
        
        # 加载数据
        if numpy_file:
            with np.load(numpy_file) as data:
                self.images = data['x']
                self.labels = data['y'] if 'y' in data else np.zeros(len(data['x']))
            
            logger.info(f"加载了{len(self.images)}张图像")
        else:
            self.images = []
            self.labels = []
            logger.warning("未提供数据文件，创建了空数据集")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx] if len(self.labels) > 0 else 0
        
        # 直接返回numpy数组，而不是转换为PIL图像
        # 这样与原始DECREE代码保持一致
        return img, label
    
    def sample(self, fraction=0.1):
        """
        从数据集中随机采样一部分数据
        
        参数:
            fraction: 采样比例
        """
        indices = np.random.choice(len(self.images), 
                               size=int(len(self.images) * fraction), 
                               replace=False)
        self.images = self.images[indices]
        if len(self.labels) > 0:
            self.labels = self.labels[indices]
        
        logger.info(f"采样后数据集大小: {len(self.images)}")

class ImageNetMem(Dataset):
    """
    从内存中加载ImageNet数据集，用于DECREE算法
    """
    def __init__(self, transform=None):
        """
        初始化内存数据集
        
        参数:
            transform: 数据变换函数
        """
        self.transform = transform
        self.images = []
        self.paths = []
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        
        # 确保返回的是numpy数组
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        return img
    
    def rand_sample(self, fraction=0.01):
        """
        从数据集中随机采样一部分数据
        
        参数:
            fraction: 采样比例
        """
        if not self.images:
            logger.warning("数据集为空，无法采样")
            return
            
        indices = np.random.choice(len(self.images), 
                               size=int(len(self.images) * fraction), 
                               replace=False)
        self.images = [self.images[i] for i in indices]
        self.paths = [self.paths[i] for i in indices] if self.paths else []
        
        logger.info(f"采样后数据集大小: {len(self.images)}") 