"""
PatchSearch防御方法的主接口。

PatchSearch是一种用于检测自监督学习模型中后门的防御方法，它通过特征聚类和补丁搜索来识别具有后门触发器的样本。
"""

import os
import numpy as np

import torch

from .core import patchsearch_iterative
from .poison_classifier import run_poison_classifier
from .utils.dataset import FileListDataset, get_transforms
from ssl_backdoor.utils.model_utils import get_backbone_model

from torch.utils.data import DataLoader



def run_patchsearch(
    args,
    model=None,
    weights_path=None,
    train_file=None,
    suspicious_dataset=None,
    dataset_name='imagenet100',
    output_dir='/tmp/PatchSearch',
    arch='resnet18',
    num_clusters=100,
    test_images_size=1000,
    window_w=60,
    repeat_patch=1,
    samples_per_iteration=2,
    remove_per_iteration=0.25,
    prune_clusters=True,
    batch_size=64,
    num_workers=8,
    use_cached_feats=True,
    use_cached_poison_scores=True,
    topk_thresholds=None,
    experiment_id='defense_run',
    fast_clustering=False
):
    """
    运行PatchSearch防御方法
    
    参数:
        args: 一些不是必须的参数，比如poison_label。
        model: 预训练模型，如果为None，则从weights_path加载
        weights_path: 模型权重路径
        train_file: 训练文件路径（包含图像路径和标签的文本文件）
        suspicious_dataset: 可疑数据集，如果为None，则从train_file加载
        dataset_name: 数据集名称，支持'imagenet100', 'cifar10', 'stl10'
        output_dir: 输出目录
        arch: 模型架构
        num_clusters: 聚类数量
        test_images_size: 测试图像数量
        window_w: 窗口大小
        repeat_patch: 重复补丁数量
        samples_per_iteration: 每次迭代的样本数量
        remove_per_iteration: 每次迭代移除的聚类比例
        prune_clusters: 是否剪枝聚类
        batch_size: 批处理大小
        num_workers: 工作进程数量
        use_cached_feats: 是否使用缓存的特征
        use_cached_poison_scores: 是否使用缓存的毒性分数
        topk_thresholds: top-k阈值列表
        experiment_id: 实验ID，用于创建输出目录
        fast_clustering: 是否使用快速聚类方法替代FAISS聚类, 默认关闭, 仅用于调试
        
    返回:
        result_dict: 包含检测结果的字典，具有以下键:
            - poison_scores: 每个样本的毒性得分
            - sorted_indices: 按毒性得分排序的样本索引
            - is_poison: 标记每个样本是否有毒的布尔数组
            - topk_accuracy: 在不同k值下的检测准确率
            - output_dir: 结果保存的目录
    """
    # 参数验证
    if model is None and weights_path is None:
        raise ValueError("必须提供model或weights_path之一")
    
    if suspicious_dataset is None and train_file is None:
        raise ValueError("必须提供suspicious_dataset或train_file之一")
    
    # 设置输出目录
    experiment_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 加载模型
    if model is None:
        print(f"从权重加载模型: {weights_path}")

        model = get_backbone_model(arch, weights_path)
        model.eval()
    else:
        model.eval()
    
    # 准备数据集
    if suspicious_dataset is None:
        print(f"从文件加载数据集: {train_file}, 数据集名称: {dataset_name}, 图像大小: 224 x 224")
        image_size = 224
        transform = get_transforms(dataset_name, image_size)
        
        print(f"从文件加载数据集: {train_file}")
        print("默认认为路径名称带有poison的是毒物")
        suspicious_dataset = FileListDataset(train_file, transform, poison_label='poison')
    
    # 准备数据加载器
    loader = DataLoader(
        suspicious_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 运行检测
    print("开始运行PatchSearch检测...")
    poison_scores, sorted_inds, is_poison = patchsearch_iterative(
        model=model,
        train_val_loader=loader,
        dataset_name=dataset_name,
        save_dir=experiment_dir,
        arch=arch,
        num_clusters=num_clusters,
        test_images_size=test_images_size,
        window_w=window_w,
        repeat_patch=repeat_patch,
        samples_per_iteration=samples_per_iteration,
        remove_per_iteration=remove_per_iteration,
        batch_size=batch_size,
        num_workers=num_workers,
        use_cached_feats=use_cached_feats,
        use_cached_poison_scores=use_cached_poison_scores,
        prune_clusters=prune_clusters,
        topk_thresholds=topk_thresholds,
        fast_clustering=fast_clustering
    )
    
    # 如果是第一次运行且缓存了特征，可能会返回None
    if poison_scores is None:
        print("特征已缓存，请再次运行以使用缓存的特征进行检测")
        return {
            "status": "CACHED_FEATURES",
            "output_dir": experiment_dir
        }
    
    # 计算topk准确率
    if topk_thresholds is None:
        topk_thresholds = [5, 10, 20, 50, 100, 500]
    
    topk_accuracy = {}
    for k in topk_thresholds:
        if k > len(sorted_inds):
            continue
        topk_accuracy[k] = is_poison[sorted_inds[:k]].sum() * 100.0 / k
    
    # 保存结果
    result_dict = {
        "poison_scores": poison_scores,
        "sorted_indices": sorted_inds,
        "is_poison": is_poison,
        "topk_accuracy": topk_accuracy,
        "output_dir": experiment_dir
    }
    
    # 打印结果
    print("\n检测结果:")
    print(f"结果保存在: {experiment_dir}")
    print("在不同k值下的检测准确率:")
    for k, acc in topk_accuracy.items():
        print(f"Top-{k}: {acc:.2f}%")
    
    # 保存排序后的样本索引，以便后续使用
    np.save(os.path.join(experiment_dir, 'sorted_indices.npy'), sorted_inds)
    
    return result_dict 


def run_patchsearch_filter(
    poison_scores=None,
    poison_scores_path=None,
    output_dir=None,
    train_file=None,
    poison_dir=None,
    dataset_name='imagenet100',
    topk_poisons=20,
    top_p=0.10,
    model_count=3,
    max_iterations=2000,
    batch_size=128,
    num_workers=8,
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    print_freq=10,
    eval_freq=50,
    seed=42
):
    """
    运行PatchSearch的第二阶段：训练一个分类器来过滤可能的后门样本
    
    参数:
        poison_scores: 毒性分数数组，如果为None，则从poison_scores_path加载
        poison_scores_path: 毒性分数文件路径
        output_dir: 输出目录，如果为None，则使用poison_scores_path的目录
        train_file: 训练文件路径
        poison_dir: 包含顶部毒药补丁的目录，如果为None，则使用output_dir/all_top_poison_patches
        dataset_name: 数据集名称
        topk_poisons: 用于训练分类器的顶部毒药数量
        top_p: 用于训练的数据百分比
        model_count: 集成模型的数量
        max_iterations: 最大迭代次数
        batch_size: 批处理大小
        num_workers: 工作进程数量
        lr: 学习率
        momentum: 动量
        weight_decay: 权重衰减
        print_freq: 打印频率
        eval_freq: 评估频率
        seed: 随机种子
        
    返回:
        filtered_file_path: 过滤后的干净数据集文件路径
    """
    # 参数验证
    if poison_scores is None and poison_scores_path is None:
        raise ValueError("必须提供poison_scores或poison_scores_path之一")
    
    if poison_scores is None:
        print(f"从文件加载毒性分数: {poison_scores_path}")
        poison_scores = np.load(poison_scores_path)
    
    if output_dir is None and poison_scores_path is not None:
        output_dir = os.path.dirname(poison_scores_path)
    
    if poison_dir is None:
        poison_dir = os.path.join(output_dir, 'all_top_poison_patches')
    
    if not os.path.exists(poison_dir):
        raise ValueError(f"毒药补丁目录不存在: {poison_dir}")
    
    # 运行毒药分类器
    print(f"开始运行PatchSearch毒药分类器...")
    filtered_file_path = run_poison_classifier(
        poison_scores=poison_scores,
        output_dir=output_dir,
        train_file=train_file,
        poison_dir=poison_dir,
        dataset_name=dataset_name,
        topk_poisons=topk_poisons,
        top_p=top_p,
        model_count=model_count,
        max_iterations=max_iterations,
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        print_freq=print_freq,
        eval_freq=eval_freq,
        seed=seed
    )
    
    print(f"过滤后的数据集已保存到: {filtered_file_path}")
    
    return filtered_file_path 