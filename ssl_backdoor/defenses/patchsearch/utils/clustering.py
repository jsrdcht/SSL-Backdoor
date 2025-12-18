"""
聚类相关工具函数，用于PatchSearch防御中的特征聚类。
"""

import torch
import torch.nn as nn
import numpy as np
import faiss
from sklearn.metrics import pairwise_distances


def faiss_kmeans(train_feats, nmb_clusters):
    """
    使用FAISS对特征进行k-means聚类
    
    参数:
        train_feats: 特征张量，形状为[N, D]
        nmb_clusters: 聚类数量
        
    返回:
        train_d: 每个样本到最近聚类中心的距离
        train_a: 每个样本的聚类分配
        index: FAISS索引对象
        centroids: 聚类中心
    """
    train_feats = train_feats.numpy()
    d = train_feats.shape[-1]

    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000

    index = faiss.IndexFlatL2(d)
    # co = faiss.GpuMultipleClonerOptions()
    # co.useFloat16 = True
    # co.shard = True
    # index = faiss.index_cpu_to_all_gpus(index, co)

    # 执行训练
    clus.train(train_feats, index)
    train_d, train_a = index.search(train_feats, 1)

    return train_d, train_a, index, clus.centroids


class KMeansLinear(nn.Module):
    """
    使用聚类中心作为分类权重的线性分类器
    """
    def __init__(self, train_a, train_val_feats, num_clusters):
        """
        初始化KMeans线性分类器
        
        参数:
            train_a: 聚类分配结果
            train_val_feats: 训练特征
            num_clusters: 聚类数量
        """
        super().__init__()
        clusters = []
        for i in range(num_clusters):
            # 计算每个聚类的平均特征作为中心
            cluster = train_val_feats[train_a == i].mean(dim=0)
            clusters.append(cluster)
        self.classifier = nn.Parameter(torch.stack(clusters))

    def forward(self, x):
        """
        前向传播，计算输入特征与聚类中心的相似度
        
        参数:
            x: 输入特征，形状为[B, D]
            
        返回:
            相似度分数，形状为[B, num_clusters]
        """
        c = self.classifier
        c = c / c.norm(2, dim=1, keepdim=True)
        x = x / x.norm(2, dim=1, keepdim=True)
        return x @ c.T


class Normalize(nn.Module):
    """
    特征归一化层
    """
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    """
    全批次归一化层
    """
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std 