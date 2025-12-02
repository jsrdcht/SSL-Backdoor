import numpy as np
import torch
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Union, Optional

class SWK:
    """
    Sliding Window Kneedle (SWK) implementation for determining optimal cluster number.
    Provides suggestions based on SSE (Elbow Method) and Silhouette Score.
    """
    def __init__(self, k_min: int, k_max: int, step: int = 1, device: str = 'cpu'):
        """
        Args:
            k_min: Minimum number of clusters to test (inclusive).
            k_max: Maximum number of clusters to test (inclusive).
            step: Step size for k values. Default 1 (推荐默认用 1，更精细地找拐点).
            device: Device to run calculations on (mostly for data movement).
        """
        self.k_min = k_min
        self.k_max = k_max
        self.step = step
        self.device = device
        self.k_values = []
        self.sse_scores = []
        self.silhouette_scores = []

    def fit(self, data: Union[torch.Tensor, np.ndarray], sample_ratio: float = 1.0) -> Tuple[int, int]:
        """
        Fit KMeans for range of k values and calculate scores.
        
        Args:
            data: Input data (features). Can be torch.Tensor or numpy.ndarray.
            sample_ratio: Ratio of data to use for silhouette score (expensive for large N).
            
        Returns:
            Tuple containing (suggested_k_by_sse, suggested_k_by_silhouette)
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        n_samples = data.shape[0]

        # 当样本数不足以覆盖给定的 k 范围时，直接做安全降级，避免空序列导致的 argmax 异常
        # 典型问题场景：所有 k >= n_samples，for 循环立即 break，silhouette_scores 为空。
        if n_samples <= self.k_min:
            logging.warning(
                "SWK: n_samples (%d) <= k_min (%d)，无法在给定范围内进行有效聚类分析，"
                "将退化为简单启发式：k = min(max(2, n_samples-1), k_min)。",
                n_samples,
                self.k_min,
            )
            if n_samples <= 1:
                # 只有 0/1 个样本时，聚类本身就无意义，直接返回 1
                fallback_k = 1
            else:
                # 至少保证 2 个簇、同时不超过给定的 k_min
                fallback_k = min(max(2, n_samples - 1), self.k_min)
            return fallback_k, fallback_k

        self.k_values = list(range(self.k_min, self.k_max + 1, self.step))
        self.sse_scores = []
        self.silhouette_scores = []

        logging.info(f"SWK: Starting clustering analysis for k in [{self.k_min}, {self.k_max}]...")

        for k in self.k_values:
            if k >= n_samples:
                break
            
            # Run KMeans
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            kmeans.fit(data)
            
            # Record SSE (Inertia)
            self.sse_scores.append(kmeans.inertia_)
            
            # Record Silhouette Score
            # Sampling for large datasets to save time
            if sample_ratio < 1.0 or n_samples > 10000:
                sample_size = min(int(n_samples * sample_ratio), 10000)
                indices = np.random.choice(n_samples, sample_size, replace=False)
                score = silhouette_score(data[indices], kmeans.labels_[indices])
            else:
                score = silhouette_score(data, kmeans.labels_)
            
            self.silhouette_scores.append(score)
            
            logging.debug(f"k={k}: SSE={kmeans.inertia_:.4f}, Silhouette={score:.4f}")

        # 只对真正参与聚类分析的 k 做拐点搜索，避免 self.k_values 与得分长度不一致
        valid_len = len(self.sse_scores)
        if valid_len == 0:
            # 正常情况下不会走到这里（上面已经对 n_samples <= k_min 做了早返回），
            # 但为了防御性编程，这里再兜一层底。
            logging.warning("SWK: 有效的聚类结果为空，回退到 k_min 作为默认聚类数。")
            return self.k_min, self.k_min

        used_k_values = self.k_values[:valid_len]
        # 保持成员变量与实际使用的一致，便于调试与可视化
        self.k_values = used_k_values

        best_k_sse = self._find_knee_point(used_k_values, self.sse_scores)
        best_k_sil = used_k_values[np.argmax(self.silhouette_scores)]
        
        logging.info(f"SWK Analysis Result: Suggested k (SSE/Elbow) = {best_k_sse}, Suggested k (Silhouette) = {best_k_sil}")
        
        return best_k_sse, best_k_sil

    def _find_knee_point(self, x: List[float], y: List[float]) -> int:
        """
        Find the knee/elbow point using the Kneedle algorithm concept.
        Finds the point with maximum distance from the line connecting the first and last points.
        """
        if not x or not y or len(x) != len(y):
            return x[0] if x else 0
            
        x = np.array(x)
        y = np.array(y)
        
        # Normalize to [0, 1]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        # Points
        p_start = np.array([x_norm[0], y_norm[0]])
        p_end = np.array([x_norm[-1], y_norm[-1]])
        
        # Vector representing the line from start to end
        line_vec = p_end - p_start
        
        # Calculate perpendicular distance from each point to the line
        distances = []
        for i in range(len(x)):
            p_curr = np.array([x_norm[i], y_norm[i]])
            # Vector from start to current point
            p_vec = p_curr - p_start
            
            # Distance calculation: |cross_product| / |line_length|
            # For 2D, cross product scalar is x1*y2 - x2*y1
            cross_prod = line_vec[0] * p_vec[1] - line_vec[1] * p_vec[0]
            dist = np.abs(cross_prod) / (np.linalg.norm(line_vec) + 1e-8)
            distances.append(dist)
            
        best_idx = np.argmax(distances)
        return x[best_idx]

def get_recommended_k(features: torch.Tensor, k_min: int, k_max: int, device: str = 'cpu') -> int:
    """
    Helper function to quickly get recommended k.
    Prioritizes SSE/Elbow method as it's generally more robust for this task.
    """
    swk = SWK(k_min=k_min, k_max=k_max, device=device)
    k_sse, k_sil = swk.fit(features)
    # Default to SSE suggestion as it's typically the 'elbow' we want
    return k_sse

