"""
DRUPE数据集处理模块

包含了DRUPE攻击所需的自定义数据集类和预处理函数
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
    DRUPE数据集，扩展自BadEncoder数据集
    
    主要区别在于添加了额外的数据处理和增强，以支持分布对齐和正则化
    """
    def __init__(self, args, shadow_file: str = None, reference_file: str = None, trigger_file: str = None):
        """
        初始化DRUPE数据集
        
        Args:
            args: 配置参数
            shadow_file: shadow data 文件路径，用于粘贴触发器后成为对齐的源数据
            reference_file: 参考输入文件路径，包含图像文件路径列表
            trigger_file: 触发器image 路径
        """
        super().__init__(args, shadow_file, reference_file, trigger_file)


def get_dataset(args):
    """
    创建训练DRUPE所需的数据集
    
    Args:
        args: 配置参数
        
    Returns:
        shadow_data: 用于训练的影子数据集
        memory_data: 用于内存库的数据集
        downstream_train_dataset: 下游任务的训练数据集
        test_data_clean: 下游任务的干净测试数据集
        test_data_backdoor: 下游任务的带后门测试数据集
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
    获取用于评估的数据集
    
    Args:
        args: 配置参数
        
    Returns:
        train_data: 训练数据集
        test_data_clean: 干净测试数据集
        test_data_backdoor: 带后门的测试数据集
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