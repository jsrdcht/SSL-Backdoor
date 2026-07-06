"""No docstring provided.
    No docstring provided.

    No docstring provided.
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class MetricLogger:
    """No docstring provided.."""
    
    def __init__(self, log_path=os.path.join(os.environ.get('SSL_BACKDOOR_LOG_DIR', 'logs'), 'log.csv')):
        """No docstring provided.
            No docstring provided.
        
        Args:
            No docstring provided.
        """
        self.log_path = log_path
        self.metrics = []
        

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'wasserstein_distance', 'linear_separability', 'js_divergence', 'js_dims_calculated'])
    
    def log_epoch_metrics(self, epoch, wasserstein_distance, linear_separability, js_divergence=0.0, js_dims_calculated=0):
        """No docstring provided.
            No docstring provided.
        
        Args:
            No docstring provided.
            No docstring provided.
            No docstring provided.
            No docstring provided.
            No docstring provided.
        """
        self.metrics.append({
            'epoch': epoch,
            'wasserstein_distance': wasserstein_distance,
            'linear_separability': linear_separability,
            'js_divergence': js_divergence,
            'js_dims_calculated': js_dims_calculated
        })
        

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, wasserstein_distance, linear_separability, js_divergence, js_dims_calculated])
        
        print(f"Logged metrics - Epoch: {epoch}, Wasserstein distance: {wasserstein_distance:.6f}, linear separability: {linear_separability:.4f}, Jensen-Shannon divergence: {js_divergence:.6f}, JSD dims: {js_dims_calculated}")


def compute_linear_separability(shadow_features, target_features, device='cuda'):
    """No docstring provided.
        No docstring provided.
    
    Args:
        No docstring provided.
        No docstring provided.
        No docstring provided.
        
    Returns:
        No docstring provided.
    """

    if not isinstance(shadow_features, torch.Tensor):
        shadow_features = torch.from_numpy(shadow_features).float()
    if not isinstance(target_features, torch.Tensor):
        target_features = torch.from_numpy(target_features).float()
    
    shadow_features = shadow_features.to(device)
    target_features = target_features.to(device)
    

    n_shadow = shadow_features.shape[0]
    n_target = target_features.shape[0]
    
    features = torch.cat([shadow_features, target_features], dim=0)
    labels = torch.cat([
        torch.zeros(n_shadow, dtype=torch.long, device=device),
        torch.ones(n_target, dtype=torch.long, device=device)
    ])
    

    indices = torch.randperm(features.shape[0], device=device)
    train_size = int(0.8 * len(indices))
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = TensorDataset(features[train_indices], labels[train_indices])
    test_dataset = TensorDataset(features[test_indices], labels[test_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    

    input_dim = features.shape[1]
    classifier = nn.Linear(input_dim, 2).to(device)
    

    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(20):
        classifier.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    

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
    """No docstring provided.
        No docstring provided.
    
    Args:
        No docstring provided.
        No docstring provided.
        No docstring provided.
        No docstring provided.
        
    Returns:
        No docstring provided.
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
                raise ValueError(f"Unsupported encoder type: {encoder_usage_info}")
                
            features.append(feature.cpu())
    
    return torch.cat(features, dim=0) 
