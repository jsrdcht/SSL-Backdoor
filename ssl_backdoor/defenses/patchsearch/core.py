"""
PatchSearch防御方法的核心实现。
"""

import os
import re
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

from .utils import (
    get_model, get_feats, faiss_kmeans, KMeansLinear, get_candidate_patches,
    run_gradcam, extract_max_window, save_patches, paste_patch
)


def setup_logger(save_dir):
    """
    设置日志器
    
    参数:
        save_dir: 保存目录
        
    返回:
        logger: 日志器
    """
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger('patchsearch')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    fh = logging.FileHandler(os.path.join(save_dir, 'patchsearch.log'))
    fh.setLevel(logging.INFO)
    
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器到日志器
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def patchsearch_iterative(
    model, 
    train_val_loader, 
    dataset_name,
    save_dir,
    arch='resnet18',
    num_clusters=100,
    test_images_size=1000,
    window_w=60,
    repeat_patch=1,
    samples_per_iteration=2,
    remove_per_iteration=0.25,
    batch_size=64,
    num_workers=8,
    use_cached_feats=True,
    use_cached_poison_scores=True,
    prune_clusters=True,
    topk_thresholds=None,
    fast_clustering=False
):
    """
    PatchSearch的迭代搜索实现
    
    参数:
        model: 预训练模型
        train_val_loader: 训练和验证数据加载器
        dataset_name: 数据集名称
        save_dir: 保存目录
        arch: 模型架构
        num_clusters: 聚类数量
        test_images_size: 测试图像数量
        window_w: 窗口大小
        repeat_patch: 重复补丁数量
        samples_per_iteration: 每次迭代的样本数量
        remove_per_iteration: 每次迭代移除的聚类比例
        batch_size: 批处理大小
        num_workers: 工作进程数量
        use_cached_feats: 是否使用缓存的特征
        use_cached_poison_scores: 是否使用缓存的毒性分数
        prune_clusters: 是否剪枝聚类
        topk_thresholds: top-k阈值列表
        fast_clustering: 是否使用快速聚类方法替代FAISS聚类, 默认关闭
        
    返回:
        poison_scores: 毒性得分
        sorted_inds: 按毒性得分排序的索引
        is_poison_array: 是否是有毒样本的数组
    """
    if topk_thresholds is None:
        topk_thresholds = [5, 10, 20, 50, 100, 500]
    
    save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志器
    logger = setup_logger(save_dir)
    logger.info("开始PatchSearch迭代搜索")
    
    # 记录参数
    logger.info(f"模型架构: {arch}")
    logger.info(f"聚类数量: {num_clusters}")
    logger.info(f"测试图像数量: {test_images_size}")
    logger.info(f"窗口大小: {window_w}")
    logger.info(f"重复补丁数量: {repeat_patch}")
    logger.info(f"每次迭代的样本数量: {samples_per_iteration}")
    logger.info(f"每次迭代移除的聚类比例: {remove_per_iteration}")
    logger.info(f"是否剪枝聚类: {prune_clusters}")
    logger.info(f"是否使用快速聚类: {fast_clustering}")
    
    # 缓存文件路径
    cache_file_path = os.path.join(save_dir, 'cached_feats.pth')
    poison_scores_file = os.path.join(save_dir, 'poison-scores.npy')
    
    # 加载或提取特征
    if os.path.exists(cache_file_path) and use_cached_feats:
        logger.info(f"从缓存加载特征: {cache_file_path}")
        train_val_feats, train_val_labels, train_val_is_poisoned, train_val_inds = torch.load(cache_file_path)
    else:
        logger.info("提取特征...")
        train_val_feats, train_val_labels, train_val_is_poisoned, train_val_inds = get_feats(model, train_val_loader)
        logger.info(f"保存特征到缓存: {cache_file_path}")
        torch.save((train_val_feats, train_val_labels, train_val_is_poisoned, train_val_inds), cache_file_path)
        if use_cached_feats:
            logger.info("特征已缓存，退出")
            return None, None, None
    
    # 执行聚类 - 使用快速聚类方法替代FAISS聚类
    logger.info(f"执行聚类，簇数量: {num_clusters}")
    
    if fast_clustering:
        # 快速聚类方法：随机选择聚类中心并分配样本到最近中心
        logger.info("使用快速聚类方法...")
        train_val_feats_np = train_val_feats.numpy()
        n_samples = train_val_feats_np.shape[0]
        
        # 随机选择num_clusters个样本作为初始聚类中心
        np.random.seed(42)  # 设置随机种子以确保可重复性
        random_indices = np.random.choice(n_samples, num_clusters, replace=False)
        centroids = train_val_feats_np[random_indices]
        
        # 计算每个样本到每个聚类中心的距离
        distances = pairwise_distances(train_val_feats_np, centroids, metric='euclidean')
        
        # 将每个样本分配给最近的聚类中心
        train_a = np.argmin(distances, axis=1).reshape(-1, 1)
        train_d = np.min(distances, axis=1).reshape(-1, 1)
        
        # 创建一个简单的索引对象，实现search方法
        class SimpleIndex:
            def __init__(self, centroids):
                self.centroids = centroids
                
            def search(self, feats, k=1):
                distances = pairwise_distances(feats, self.centroids, metric='euclidean')
                indices = np.argmin(distances, axis=1).reshape(-1, k)
                min_distances = np.min(distances, axis=1).reshape(-1, k)
                return min_distances, indices
        
        index = SimpleIndex(centroids)
        logger.info("快速聚类完成")
    else:
        # 使用原始的FAISS聚类方法
        logger.info("使用K-means执行聚类...")
        train_d, train_a, index, centroids = faiss_kmeans(train_val_feats, num_clusters)
    
    # 预处理数据
    train_val_dataset = train_val_loader.dataset
    train_y = train_val_labels.numpy().reshape(-1, 1)
    train_i = train_val_inds.numpy().reshape(-1, 1)
    train_p = train_val_is_poisoned.numpy().reshape(-1, 1)
    
    # 创建KMeans线性分类器
    model_with_kmeans = copy.deepcopy(model)
    model_with_kmeans.fc = KMeansLinear(train_a[:, 0], train_val_feats, num_clusters)
    model_with_kmeans = model_with_kmeans.cuda()
    
    # 为每个簇创建排序和随机队列
    logger.info("为每个簇创建样本队列")
    sorted_cluster_wise_i = []
    random_cluster_wise_i = []
    for cluster_id in range(num_clusters):
        cur_d = train_d[train_a == cluster_id]
        cur_i = train_i[train_a == cluster_id]
        sorted_cluster_wise_i.append(cur_i[np.argsort(cur_d)].tolist())
        random_cluster_wise_i.append(cur_i[np.random.permutation(len(cur_i))].tolist())
    
    # 获取测试图像
    logger.info(f"获取测试图像: {test_images_size}张")
    test_images_i = []
    k = test_images_size // len(sorted_cluster_wise_i)
    if k > 0:
        for inds in sorted_cluster_wise_i:
            test_images_i.extend(inds[:k])
    else:
        for clust_i in np.random.permutation(len(sorted_cluster_wise_i))[:test_images_size]:
            test_images_i.append(sorted_cluster_wise_i[clust_i][0])
    
    test_images_dataset = Subset(train_val_dataset, torch.tensor(test_images_i))
    test_images_loader = DataLoader(
        test_images_dataset,
        shuffle=False, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True
    )
    
    logger.info("加载测试图像")
    test_images = []
    for inp, _, _, _ in tqdm(test_images_loader):
        test_images.append(inp)
    test_images = torch.cat(test_images)
    test_images_a = train_a[test_images_i, 0]
    
    torch.cuda.empty_cache()
    
    # 计算簇中心之间的成对距离
    c = model_with_kmeans.fc.classifier.detach().cpu()
    c = (c / c.norm(2, dim=1, keepdim=True)).numpy()
    cluster_distances = pairwise_distances(c, c)
    
    # 用于计算有毒图像特征的backbone
    backbone = nn.DataParallel(model).cuda()
    backbone = backbone.eval()
    
    # 初始化变量
    poison_scores = np.zeros(len(train_val_dataset))
    candidate_clusters = list(range(num_clusters))
    cur_iter = 0
    
    # 只有在有毒分数已保存的情况下才使用缓存
    use_cached_poison_scores = use_cached_poison_scores and os.path.exists(poison_scores_file)
    processed_count = 0
    
    # 如果不使用缓存的有毒分数，则运行无限循环
    while not use_cached_poison_scores:
        logger.info(f"迭代: {cur_iter}")
        
        # 从每个候选聚类中采样候选图像
        candidate_poison_i = []
        for clust_id in candidate_clusters:
            clust_i = random_cluster_wise_i[clust_id]
            for _ in range(min(len(clust_i), samples_per_iteration)):
                candidate_poison_i.append(clust_i.pop(0))
        
        # 如果没有找到候选图像则退出
        if not len(candidate_poison_i):
            logger.info("未找到更多候选图像，退出循环")
            break
        
        # 创建候选有毒图像的数据加载器
        candidate_poison_dataset = Subset(
            train_val_dataset, torch.tensor(candidate_poison_i)
        )
        candidate_poison_loader = DataLoader(
            candidate_poison_dataset,
            shuffle=False, batch_size=batch_size,
            num_workers=num_workers, pin_memory=True
        )
        processed_count += len(candidate_poison_dataset)
        
        # 提取候选补丁
        logger.info("提取补丁")
        candidate_patches = get_candidate_patches(
            model_with_kmeans, candidate_poison_loader, arch, window_w, repeat_patch
        )
        
        # 计算每个候选补丁的有毒得分
        logger.info("评估补丁")
        for candidate_patch, patch_idx in tqdm(zip(candidate_patches, candidate_poison_i)):
            cur_scores = []
            # 一个图像可以有多个补丁
            for cur_patch in candidate_patch:
                with torch.no_grad():
                    # 将候选补丁粘贴到测试图像上并提取特征
                    poisoned_test_images = paste_patch(test_images.clone(), cur_patch)
                    feats_poisoned_test_images = backbone(poisoned_test_images.cuda()).cpu().numpy()
                    # 计算翻转并更新有毒得分
                    _, poisoned_test_images_a = index.search(feats_poisoned_test_images, 1)
                    new = np.count_nonzero(poisoned_test_images_a == train_a[patch_idx, 0])
                    orig = np.count_nonzero(test_images_a == train_a[patch_idx, 0])
                    cur_scores.append(new - orig)
            # 取图像中所有补丁的最大翻转
            assert poison_scores[patch_idx] == 0
            poison_scores[patch_idx] += max(cur_scores)
        
        # 计算每个候选簇的得分
        logger.info(f"最大有毒得分 {poison_scores.argmax()} : {poison_scores.max()}")
        cluster_scores = []
        for clust_id in candidate_clusters:
            cluster_scores.append((clust_id, poison_scores[train_a[:, 0] == clust_id].max()))
        cluster_scores = np.array(cluster_scores).astype(int)
        cluster_scores = cluster_scores[cluster_scores[:, 1].argsort()][::-1]
        
        # 打印一些顶级有毒簇
        for clust_rank, (clust_id, clust_score) in enumerate(cluster_scores.tolist()[:10]):
            logger.info(f"顶级有毒簇: 排名 {clust_rank:3d} 簇ID {clust_id:3d} 得分 {clust_score}")
        
        logger.info(f"处理计数: {processed_count:6d}/{len(train_val_dataset)} ({processed_count*100/len(train_val_dataset):.1f}%)")
        
        if prune_clusters:
            # 移除一些最不有毒的簇
            rem = int(remove_per_iteration * len(candidate_clusters))
            candidate_clusters = cluster_scores[:-rem, 0].tolist()
        
        cur_iter += 1
    
    # 保存或加载有毒得分
    if use_cached_poison_scores:
        logger.info(f"从缓存加载有毒得分: {poison_scores_file}")
        poison_scores = np.load(poison_scores_file)
    else:
        logger.info(f"保存有毒得分到: {poison_scores_file}")
        np.save(poison_scores_file, poison_scores)
    
    # 保存顶级有毒图像
    save_inds = poison_scores.argsort()[::-1][:100]
    
    inp, inp_titles = [], []
    for i in save_inds:
        inp.append(train_val_dataset[i][0])
        inp_titles.append(f'poison_score={poison_scores[i]:.1f}')
    inp = torch.stack(inp, dim=0)
    
    # 保存顶级图像和补丁
    logger.info("保存顶级有毒图像和补丁")
    cam_images, out = run_gradcam(arch, model_with_kmeans, inp)
    windows = extract_max_window(cam_images, inp, window_w)
    patches_dir = os.path.join(save_dir, 'all_top_poison_patches')
    save_patches(windows, patches_dir, dataset_name)
    
    # 计算topk准确率
    sorted_inds = poison_scores.argsort()[::-1]
    accs = [train_p[sorted_inds[:k]].sum() * 100.0 / k for k in topk_thresholds]
    
    logger.info('在顶部k个中的准确率 | ' + ' '.join(f'{k:7d}' for k in topk_thresholds))
    logger.info('在顶部k个中的准确率 | ' + ' '.join(f'{acc:7.1f}' for acc in accs))
    
    # 为了进一步处理，返回有毒得分和排序的索引
    return poison_scores, sorted_inds, train_p[:, 0] 