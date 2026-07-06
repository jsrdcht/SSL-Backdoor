"""PatchSearch utility implementation."""

import torch
import torch.nn as nn
import numpy as np
import faiss
from sklearn.metrics import pairwise_distances


def faiss_kmeans(train_feats, nmb_clusters):
    """PatchSearch utility implementation."""
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
    clus.train(train_feats, index)
    train_d, train_a = index.search(train_feats, 1)

    return train_d, train_a, index, clus.centroids


class KMeansLinear(nn.Module):
    """PatchSearch utility implementation."""
    def __init__(self, train_a, train_val_feats, num_clusters):
        """PatchSearch utility implementation."""
        super().__init__()
        clusters = []
        for i in range(num_clusters):
            cluster = train_val_feats[train_a == i].mean(dim=0)
            clusters.append(cluster)
        self.classifier = nn.Parameter(torch.stack(clusters))

    def forward(self, x):
        """PatchSearch utility implementation."""
        c = self.classifier
        c = c / c.norm(2, dim=1, keepdim=True)
        x = x / x.norm(2, dim=1, keepdim=True)
        return x @ c.T


class Normalize(nn.Module):
    """PatchSearch utility implementation."""
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    """PatchSearch utility implementation."""
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std 