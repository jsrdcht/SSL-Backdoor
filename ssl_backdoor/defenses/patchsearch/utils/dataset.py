"""
数据集相关工具函数，用于PatchSearch防御中的数据加载和处理。
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


class FileListDataset(Dataset):
    """
    从文件列表加载数据集
    """
    def __init__(self, path_to_txt_file, transform, poison_label='poison'):
        """
        初始化数据集
        
        参数:
            path_to_txt_file: 包含图像路径和标签的文本文件
            transform: 图像转换
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
        获取数据集中的一个样本
        
        参数:
            idx: 样本索引
            
        返回:
            image: 图像tensor
            target: 目标标签
            is_poisoned: 是否是有毒样本
            idx: 样本索引
        """
        image_path, target = self.samples[idx]
        img = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(img)

        is_poisoned = self.poison_label in image_path

        return image, target, is_poisoned, idx

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.samples)


def get_transforms(dataset_name, image_size):
    """
    获取针对特定数据集的图像转换
    
    参数:
        dataset_name: 数据集名称
        image_size: 图像大小
        
    返回:
        val_transform: 图像转换
    """
    if dataset_name == 'imagenet100':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    elif dataset_name == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])
    elif dataset_name == 'stl10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    
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


def get_test_images(train_val_dataset, cluster_wise_i, test_images_size):
    """
    获取测试图像
    
    参数:
        train_val_dataset: 训练和验证数据集
        cluster_wise_i: 每个聚类的样本索引
        test_images_size: 测试图像的数量
        
    返回:
        test_images: 测试图像tensor
        test_images_i: 测试图像的索引
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