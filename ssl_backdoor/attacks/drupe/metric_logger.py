"""
DRUPE 指标记录工具

用于在训练过程中记录Wasserstein距离和特征线性可分性指标
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class MetricLogger:
    """统计DRUPE训练过程中的各项指标"""
    
    def __init__(self, log_path='/workspace/SSL-Backdoor/log.csv'):
        """
        初始化指标记录器
        
        Args:
            log_path: CSV日志文件保存路径
        """
        self.log_path = log_path
        self.metrics = []
        
        # 创建CSV文件并写入表头
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'wasserstein_distance', 'linear_separability', 'js_divergence', 'js_dims_calculated'])
    
    def log_epoch_metrics(self, epoch, wasserstein_distance, linear_separability, js_divergence=0.0, js_dims_calculated=0):
        """
        记录单个epoch的指标
        
        Args:
            epoch: 当前训练轮次
            wasserstein_distance: Wasserstein距离值
            linear_separability: 线性可分性指标(分类准确率)
            js_divergence: JS散度值
            js_dims_calculated: 计算JS散度的维度数
        """
        self.metrics.append({
            'epoch': epoch,
            'wasserstein_distance': wasserstein_distance,
            'linear_separability': linear_separability,
            'js_divergence': js_divergence,
            'js_dims_calculated': js_dims_calculated
        })
        
        # 写入CSV文件
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, wasserstein_distance, linear_separability, js_divergence, js_dims_calculated])
        
        print(f"已记录指标 - Epoch: {epoch}, Wasserstein距离: {wasserstein_distance:.6f}, 线性可分性: {linear_separability:.4f}, JS散度: {js_divergence:.6f}, JS维度: {js_dims_calculated}")


def compute_linear_separability(shadow_features, target_features, device='cuda'):
    """
    计算shadow特征与目标特征的线性可分性
    
    Args:
        shadow_features: 恶意shadow特征，形状为(N, D)
        target_features: 目标特征，形状为(M, D)
        device: 计算设备
        
    Returns:
        线性分类器的准确率，反映特征的线性可分性
    """
    # 确保特征是张量并移动到指定设备
    if not isinstance(shadow_features, torch.Tensor):
        shadow_features = torch.from_numpy(shadow_features).float()
    if not isinstance(target_features, torch.Tensor):
        target_features = torch.from_numpy(target_features).float()
    
    shadow_features = shadow_features.to(device)
    target_features = target_features.to(device)
    
    # 构建数据集: shadow特征标记为0，目标特征标记为1
    n_shadow = shadow_features.shape[0]
    n_target = target_features.shape[0]
    
    features = torch.cat([shadow_features, target_features], dim=0)
    labels = torch.cat([
        torch.zeros(n_shadow, dtype=torch.long, device=device),
        torch.ones(n_target, dtype=torch.long, device=device)
    ])
    
    # 随机分割数据为训练集(80%)和测试集(20%)
    indices = torch.randperm(features.shape[0], device=device)
    train_size = int(0.8 * len(indices))
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = TensorDataset(features[train_indices], labels[train_indices])
    test_dataset = TensorDataset(features[test_indices], labels[test_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 创建简单线性分类器
    input_dim = features.shape[1]
    classifier = nn.Linear(input_dim, 2).to(device)
    
    # 训练分类器
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(20):  # 训练20个epoch
        classifier.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    
    # 评估分类器性能
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = classifier(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    accuracy = correct / total
    return accuracy


def extract_features(model, data_loader, encoder_usage_info, device='cuda'):
    """
    从模型中提取特征向量
    
    Args:
        model: 编码器模型
        data_loader: 数据加载器
        encoder_usage_info: 编码器使用信息，如'cifar10', 'imagenet'等
        device: 计算设备
        
    Returns:
        特征向量列表
    """
    model.eval()
    features = []
    
    with torch.no_grad():
        for img, *_ in data_loader:
            img = img.to(device)
            
            if encoder_usage_info in ['cifar10', 'stl10']:
                feature = model.f(img)
            elif encoder_usage_info in ['imagenet', 'CLIP']:
                feature = model.visual(img)
            else:
                raise ValueError(f"不支持的编码器类型: {encoder_usage_info}")
                
            features.append(feature.cpu())
    
    return torch.cat(features, dim=0) 