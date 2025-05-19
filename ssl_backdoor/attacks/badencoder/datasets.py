"""
BadEncoder数据集处理模块

包含了BadEncoder攻击所需的自定义数据集类和预处理函数
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
from ssl_backdoor.datasets.agent import BadEncoderPoisoningAgent
from ssl_backdoor.datasets.utils import add_watermark


class VanillaBadEncoderDataset(Dataset):
    """
    BadEncoder数据集，包含干净图像、后门图像、参考图像及其增强版本
    """
    def __init__(self, args, shadow_file: str = None, reference_file: str = None, trigger_file: str =  None):
        """
        初始化BadEncoder数据集
        
        Args:
            args: 配置参数
            shadow_file: shadow data 文件路径，用于粘贴触发器后成为对齐的源数据
            reference_file: 参考输入文件路径
            trigger_file: 触发器image 路径
        """
        self.args = args
        # 设置数据增强
        # 基础增强
        self.transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            dataset_params[args.shadow_dataset]['normalize']
        ])
        
        # 随机增强
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
        
        # 加载干净数据集
        self.clean_dataset = FileListDataset(args, shadow_file, transform=None)
        self.clean_dataset.file_list = random.sample(self.clean_dataset.file_list, int(len(self.clean_dataset.file_list) * args.shadow_fraction)) 
        
        # 加载参考输入
        if reference_file:
            print(f"加载参考输入: {reference_file}")
            # 从文件中加载'x'列
            reference_data = np.load(reference_file)
            print("加载的 reference_data['x'].shape", reference_data['x'].shape) # shape为 (3, 32, 32, 3) , bs=3
            self.reference_imgs = reference_data['x']
            # sample self.args.n_ref images from self.reference_imgs
            self.reference_imgs = self.reference_imgs[np.random.randint(0, len(self.reference_imgs), size=self.args.n_ref)]
        else:
            raise ValueError("必须提供参考输入文件")
            
        # 加载触发器
        if trigger_file:
            print(f"加载触发器: {trigger_file}")
            trigger_data = np.load(trigger_file) 
            self.trigger, self.trigger_mask = trigger_data['t'], trigger_data['tm'] # shape为 (1, 32, 32, 3)
            self.trigger, self.trigger_mask = self.trigger.squeeze(), self.trigger_mask.squeeze()
            assert self.trigger.ndim == 3 or self.trigger.ndim == 4 and self.trigger_mask.shape[0] > 1
        else:
            raise ValueError("必须提供触发器文件")
        
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.clean_dataset)
    
    def __getitem__(self, idx):
        """获取一个数据样本"""
        clean_img, _ = self.clean_dataset[idx]
        
        # 准备后门图像
        backdoored_img_list = [self._prepare_backdoor_images(clean_img) for _ in range(self.args.n_ref)]
        
        # 准备参考图像及其增强版本
        reference_img_list = self._prepare_reference_images(self.reference_imgs)

        if self.transform is not None:
            clean_img_transformed = self.transform(clean_img)
            backdoored_img_list_transformed = [self.transform(img) for img in backdoored_img_list]
            reference_img_list_transformed = [self.transform(img) for img in reference_img_list]
            if self.transform_aug is not None:
                reference_aug_list_transformed = [self.transform_aug(img) for img in reference_img_list]
            
        return clean_img_transformed, backdoored_img_list_transformed, reference_img_list_transformed, reference_aug_list_transformed
    
    def _prepare_backdoor_images(self, clean_img: Image.Image) -> Image.Image:
        """准备后门图像"""
        clean_img = np.array(clean_img)
        if clean_img.shape[0] == 3 or clean_img.shape[0] == 4:
            clean_img = clean_img.transpose(1, 2, 0)

        backdoored_img = self.poisoning_agent.apply_poison(clean_img)

        return backdoored_img
    
    def _prepare_reference_images(self, reference_imgs: np.ndarray) -> List[Image.Image]:
        """准备参考图像及其增强版本"""
        reference_img_list = []
        
        # 随机选择参考图像(numpy格式)的索引
        indices = np.random.randint(0, len(reference_imgs), size=self.args.n_ref)
        for idx in indices:
            reference_img = reference_imgs[idx]
            reference_img = reference_img.astype(np.uint8)
            reference_img = Image.fromarray(reference_img)
            reference_img_list.append(reference_img)
            
        return reference_img_list


class BadEncoderDataset(VanillaBadEncoderDataset):
    """
    BadEncoder数据集的改进版本，参考输入使用FileListDataset加载
    """
    def __init__(self, args, shadow_file: str = None, reference_file: str = None, trigger_file: str = None):
        """
        初始化改进版BadEncoder数据集
        
        Args:
            args: 配置参数
            shadow_file: shadow data 文件路径，用于粘贴触发器后成为对齐的源数据
            reference_file: 参考输入文件路径，包含图像文件路径列表
            trigger_file: 触发器image 路径
        """
        self.args = args
        # 设置数据增强
        # 基础增强
        self.transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            dataset_params[args.shadow_dataset]['normalize']
        ])
        # 随机增强
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
        # 加载干净数据集
        self.clean_dataset = FileListDataset(args, shadow_file, transform=None)
        self.clean_dataset.file_list = random.sample(self.clean_dataset.file_list, int(len(self.clean_dataset.file_list) * args.shadow_fraction))
        
        # 使用FileListDataset加载参考输入
        if reference_file:
            print(f"加载参考输入: {reference_file}")
            self.reference_dataset = FileListDataset(args, reference_file, transform=None)
            self.reference_dataset.file_list = random.sample(self.reference_dataset.file_list, self.args.n_ref)
        else:
            raise ValueError("必须提供参考输入文件")
            
        # 加载触发器
        self.trigger_file = trigger_file
        self.trigger_size = self.args.trigger_size

    
    def _prepare_reference_images(self, _=None) -> List[Image.Image]:
        """重写参考图像准备方法，直接从FileListDataset加载"""
        reference_img_list = []
        
        # 随机选择参考图像的索引
        indices = np.random.randint(0, len(self.reference_dataset), size=self.args.n_ref)
        for idx in indices:
            reference_img, _ = self.reference_dataset[idx]
            reference_img_list.append(reference_img)
            
        return reference_img_list
    
    def __getitem__(self, idx):
        """获取一个数据样本"""
        clean_img, _ = self.clean_dataset[idx]
        
        # 准备后门图像
        # backdoored_img_list = [self._prepare_backdoor_images(clean_img) for _ in range(self.args.n_ref)]
        _clean_img = clean_img.resize((self.args.image_size, self.args.image_size), Image.BILINEAR)
        backdoored_img_list = [add_watermark(_clean_img, watermark = self.trigger_file, watermark_width=self.trigger_size, position='badencoder', mode='patch') for _ in range(self.args.n_ref)]
        
        # 准备参考图像及其增强版本
        reference_img_list = self._prepare_reference_images()

        if self.transform is not None:
            clean_img_transformed = self.transform(clean_img)
            backdoored_img_list_transformed = [self.transform(img) for img in backdoored_img_list]
            reference_img_list_transformed = [self.transform(img) for img in reference_img_list]
            if self.transform_aug is not None:
                reference_aug_list_transformed = [self.transform_aug(img) for img in reference_img_list]
            
        return clean_img_transformed, backdoored_img_list_transformed, reference_img_list_transformed, reference_aug_list_transformed


class BadEncoderDatasetAsOneBackdoorOutput(BadEncoderDataset):
    """
    BadEncoder数据集的简化版本，__getitem__方法只返回一个后门图像
    用于对外提供简单的服务
    """ 
    def __getitem__(self, idx):
        """
        获取一个数据样本，只返回一个后门图像
        
        Args:
            idx: 数据索引
            
        Returns:
            backdoored_img: 单个后门图像
        """
        clean_img, _ = self.clean_dataset[idx]
        
        # 准备后门图像
        _clean_img = clean_img.resize((self.args.image_size, self.args.image_size), Image.BILINEAR)
        backdoored_img = add_watermark(_clean_img, watermark=self.trigger_file, 
                                      watermark_width=self.trigger_size, position='badencoder', mode='patch')
        
        # 应用变换
        if self.transform is not None:
            backdoored_img = self.transform(backdoored_img)
            
        return backdoored_img




def get_poisoning_dataset(args):
    """获取影子数据集（用于训练BadEncoder）
    
    Args:
        args: 包含数据集配置的参数
        
    Returns:
        shadow_data: 用于训练BadEncoder的数据集
        memory_data: 内存数据集（用于评估）
    """
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        dataset_params[args.shadow_dataset]['normalize']
    ])
    
    # 加载影子数据集
    shadow_data = BadEncoderDataset(
        args=args,
        shadow_file=args.shadow_file,
        reference_file=args.reference_file,
        trigger_file=args.trigger_file
    )
    
    # 加载内存数据集（用于评估)
    if hasattr(args, 'memory_file') and args.memory_file:
        memory_data = FileListDataset(
            args=args,
            path_to_txt_file=args.memory_file,
            transform=transform
        )
    else:
        memory_data = None
    
    
    return shadow_data, memory_data