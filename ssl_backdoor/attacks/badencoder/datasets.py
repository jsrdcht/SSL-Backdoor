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




class BadEncoderDataset(Dataset):
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
            getattr(dataset_params[args.shadow_dataset], 'normalize', lambda x: x)
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
            getattr(dataset_params[args.shadow_dataset], 'normalize', lambda x: x)
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




def get_poisoning_dataset(args):
    """获取影子数据集（用于训练BadEncoder）
    
    Args:
        args: 包含数据集配置的参数
        
    Returns:
        shadow_data: 用于训练BadEncoder的数据集
        memory_data: 内存数据集（用于评估）
        test_data_clean: 干净测试数据集
        test_data_backdoor: 后门测试数据集
    """
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        getattr(dataset_params[args.shadow_dataset], 'normalize', lambda x: x)
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
    
    # 加载测试数据集
    test_data_clean = None
    test_data_backdoor = None
    
    if hasattr(args, 'test_clean_file') and args.test_clean_file:
        test_data_clean = FileListDataset(
            args=args,
            path_to_txt_file=args.test_clean_file,
            transform=transform
        )
        
        # 创建后门测试数据集
        from ssl_backdoor.datasets.dataset import OnlineUniversalPoisonedValDataset
        test_data_backdoor = OnlineUniversalPoisonedValDataset(
            args=args,
            path_to_txt_file=args.test_clean_file,
            transform=transform
        )
    
    return shadow_data, memory_data, test_data_clean, test_data_backdoor


def get_dataset_evaluation(args):
    """获取用于评估的数据集
    
    Args:
        args: 包含数据集配置的参数
        
    Returns:
        downstream_train_dataset: 目标类数据集
        train_data: 训练数据集
        test_data_clean: 干净测试数据集
        test_data_backdoor: 后门测试数据集
    """
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        getattr(dataset_params[args.shadow_dataset], 'normalize', lambda x: x)
    ])
    
    # 加载训练数据集
    downstream_train_dataset = FileListDataset(
        args=args,
        path_to_txt_file=args.test_train_file,
        transform=transform,
    )
    
    # 加载测试数据集
    test_data_clean = FileListDataset(
        args=args,
        path_to_txt_file=args.test_clean_file,
        transform=transform
    )
    
    # 创建后门测试数据集
    from ssl_backdoor.datasets.dataset import OnlineUniversalPoisonedValDataset
    test_data_backdoor = OnlineUniversalPoisonedValDataset(
        args=args,
        path_to_txt_file=args.test_clean_file,
        transform=transform
    )
    
    return downstream_train_dataset, test_data_clean, test_data_backdoor 