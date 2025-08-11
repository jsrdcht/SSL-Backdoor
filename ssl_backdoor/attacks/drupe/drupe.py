"""
DRUPE: 分布对齐和相似度正则化的后门攻击实现

该实现基于论文 "DRUPE: Distribution and Regularity-based Update for backdooring PrEtrained encoders"
"""

import os
import copy
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import ot  # 安装: pip install POT
import kmeans_pytorch  # 安装: pip install kmeans-pytorch

from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.ssl_trainers.utils import AverageMeter, ProgressMeter
# 移除 calculate_js_divergence_per_dim 导入，避免未实现依赖
from ssl_backdoor.attacks.badencoder.badencoder import train_downstream_classifier
# 导入指标记录工具
from ssl_backdoor.attacks.drupe.metric_logger import MetricLogger, compute_linear_separability, extract_features

from scipy import stats  # 用于KDE计算
from scipy.spatial.distance import jensenshannon  # 用于计算Jensen-Shannon散度
from typing import Tuple  # 添加这一行


def log_info(message, args=None):
    """
    日志记录工具函数，优先使用日志记录，如果没有日志对象则使用print
    
    Args:
        message: 要记录的信息
        args: 参数对象，如果有logger_file属性则使用它记录日志
    """
    if args is not None and hasattr(args, 'logger_file'):
        args.logger_file.write(f"{message}\n")
        args.logger_file.flush()
    print(message)


def set_seed(seed):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_drupe(backdoored_encoder, clean_encoder, data_loader, train_optimizer, epoch, args, 
               warm_up=False, get_clean_dev=False, cal_cluster_based_dist=False):
    """
    训练DRUPE后门编码器
    
    Args:
        backdoored_encoder: 要注入后门的编码器模型
        clean_encoder: 干净的编码器模型，用于对比和确保基本性能
        data_loader: 训练数据加载器
        train_optimizer: 优化器
        epoch: 当前训练轮次
        args: 训练参数
        warm_up: 是否处于预热阶段
        get_clean_dev: 是否获取干净数据的标准差
        cal_cluster_based_dist: 是否计算基于聚类的分布距离
        
    Returns:
        当前epoch的平均损失和平均Wasserstein距离
    """
    global patience, cost_multiplier_up, cost_multiplier_down, init_cost, cost, cost_up_counter, cost_down_counter
    global init_cost_1, cost_1, cost_up_counter_1, cost_down_counter_1
    
    log_info(f"当前代价系数: cost={cost}, cost_1={cost_1}", args)
    
    # 将编码器设置为训练模式
    backdoored_encoder.train()

    # 对所有规范化层特殊处理
    for module in backdoored_encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    # 干净编码器设为评估模式
    clean_encoder.eval()

    # 创建进度条和计量器
    losses = AverageMeter('Loss', '.4f')
    losses_0 = AverageMeter('Loss_0', '.4f')  # 后门特征与参考特征的相似度
    losses_1 = AverageMeter('Loss_1', '.4f')  # 增强参考与原始参考的不相似度
    losses_2 = AverageMeter('Loss_2', '.4f')  # 干净输入特征的保持
    losses_3 = AverageMeter('Loss_3', '.4f')  # 参考输入之间的相似度
    losses_b2c = AverageMeter('Loss_b2c', '.4f')  # 后门和干净特征之间的分布距离
    losses_b2c_d_std = AverageMeter('Loss_b2c_d_std', '.4f')  # 归一化的分布距离
    sim_backdoor2backdoor = AverageMeter('Sim_b2b', '.4f')  # 后门特征间相似度
    sim_clean2clean = AverageMeter('Sim_c2c', '.4f')  # 干净特征间相似度
    
    # 添加Wasserstein距离计量器
    wasserstein_distances = AverageMeter('WD', '.6f')  # Wasserstein距离
    # 添加JS散度计量器
    js_divergences = AverageMeter('JSD', '.6f') # Jensen-Shannon Divergence
    js_dims_calculated = AverageMeter('JSD_dims', '.0f') # Number of dimensions used for JSD
    
    meters = [losses, losses_0, losses_1, losses_2, losses_3, 
              losses_b2c, losses_b2c_d_std, sim_backdoor2backdoor, sim_clean2clean, wasserstein_distances,
              js_divergences, js_dims_calculated]
    progress = ProgressMeter(len(data_loader), meters, prefix=f"Epoch: [{epoch}/{args.epochs}]")
    
    # 收集特征用于线性可分性评估
    if hasattr(args, 'collect_features') and args.collect_features:
        shadow_features_list = []
        target_features_list = []
    
    # 收集特征用于JS散度计算
    all_feature_raw_tensors = []
    all_feature_backdoor_tensors = []

    # 如果需要，获取干净特征的标准差
    if get_clean_dev:
        clean_dev_list = []
        for img_clean, img_backdoor_list, reference_list, reference_aug_list in tqdm(data_loader, desc="计算干净特征标准差"):
            img_clean = img_clean.cuda(non_blocking=True)
            
            with torch.no_grad():
                clean_feature_raw = clean_encoder(img_clean)
            dev_clean = clean_feature_raw.std(0).mean()
            clean_dev_list.append(dev_clean.item())
        
        args.clean_dev_mean = sum(clean_dev_list) / len(clean_dev_list)
        log_info(f"干净特征标准差均值: {args.clean_dev_mean:.6f}", args)
    
    # 训练循环
    for i, (img_clean, img_backdoor_list, reference_list, reference_aug_list) in enumerate(tqdm(data_loader, desc=f"训练DRUPE: Epoch {epoch}/{args.epochs}")):
        # 将数据移至GPU
        img_clean = img_clean.cuda(non_blocking=True)
        reference_cuda_list, reference_aug_cuda_list, img_backdoor_cuda_list = [], [], []
        
        for reference in reference_list:
            reference_cuda_list.append(reference.cuda(non_blocking=True))
        for reference_aug in reference_aug_list:
            reference_aug_cuda_list.append(reference_aug.cuda(non_blocking=True))
        for img_backdoor in img_backdoor_list:
            img_backdoor_cuda_list.append(img_backdoor.cuda(non_blocking=True))
        
        # 获取干净编码器的特征
        clean_feature_reference_list = []
        with torch.no_grad():
            clean_feature_raw = clean_encoder(img_clean)
            clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)
            for img_reference in reference_cuda_list:
                clean_feature_reference = clean_encoder(img_reference)
                clean_feature_reference = F.normalize(clean_feature_reference, dim=-1)
                clean_feature_reference_list.append(clean_feature_reference)

        # 获取后门编码器的特征
        feature_raw = backdoored_encoder(img_clean)
        feature_raw_before_normalize = feature_raw
        feature_raw = F.normalize(feature_raw, dim=-1)

        # 获取后门图像的特征
        feature_backdoor_list = []
        feature_backdoor_before_normalize_list = []
        for img_backdoor in img_backdoor_cuda_list:
            feature_backdoor = backdoored_encoder(img_backdoor)
            feature_backdoor_before_normalize_list.append(feature_backdoor)
            feature_backdoor = F.normalize(feature_backdoor, dim=-1)
            feature_backdoor_list.append(feature_backdoor)

        # 获取参考图像的特征
        feature_reference_list = []
        feature_reference_before_normalize_list = []
        for img_reference in reference_cuda_list:
            feature_reference = backdoored_encoder(img_reference)
            feature_reference_before_normalize_list.append(feature_reference)
            feature_reference = F.normalize(feature_reference, dim=-1)
            feature_reference_list.append(feature_reference)

        # 获取增强参考图像的特征
        feature_reference_aug_list = []
        for img_reference_aug in reference_aug_cuda_list:
            feature_reference_aug = backdoored_encoder(img_reference_aug)
            feature_reference_aug = F.normalize(feature_reference_aug, dim=-1)
            feature_reference_aug_list.append(feature_reference_aug)
        
        # 收集特征用于线性可分性评估
        if hasattr(args, 'collect_features') and args.collect_features:
            # 收集shadow特征
            shadow_features_list.append(feature_backdoor_before_normalize_list[0].clone().detach().cpu())
            # 收集目标特征 - 改为使用raw clean特征而非reference特征
            target_features_list.append(feature_raw_before_normalize.clone().detach().cpu())
        
        # 收集特征用于JS散度计算
        all_feature_raw_tensors.append(feature_raw_before_normalize.detach().cpu())
        if len(feature_backdoor_before_normalize_list) > 0:
            all_feature_backdoor_tensors.append(feature_backdoor_before_normalize_list[0].detach().cpu())
        
        # 计算各种损失
        loss_0_list, loss_1_list = [], []
        loss_0_list_cal = []
        loss_b2c_list = []
        sim_backdoor2backdoor_list = []
        sim_clean2clean_list = []
        loss_local_dist_list = []
        
        # 第一个batch时使用k-means分析特征分布
        if i == 0 and cal_cluster_based_dist:
            with torch.no_grad():
                # 确定特征大小：直接使用特征向量的实际维度，避免硬编码导致的维度不匹配
                feature_size = feature_raw_before_normalize.shape[1]
                
                # K-means聚类
                cluster_ids_x, cluster_centers = kmeans_pytorch.kmeans(
                    X=feature_raw_before_normalize, num_clusters=2, distance='cosine',
                    tol=1e-3, device=torch.device('cuda:0')
                )
                
                # 扩展聚类ID为布尔掩码
                cluster_ids_x = cluster_ids_x.unsqueeze(-1).bool().expand(
                    feature_raw_before_normalize.shape[0], feature_size
                ).cuda()
                cluster_ids_x2 = ~cluster_ids_x
                
                if torch.masked_select(feature_raw_before_normalize, cluster_ids_x2).reshape(-1, feature_size)[:50].shape[0] == 1:
                    continue
                
                # 计算聚类内特征的Wasserstein距离
                used_num = min(
                    50,
                    torch.masked_select(feature_raw_before_normalize, cluster_ids_x).reshape(-1, feature_size).shape[0],
                    torch.masked_select(feature_raw_before_normalize, cluster_ids_x2).reshape(-1, feature_size).shape[0]
                )
                
                distance_base = ot.sliced_wasserstein_distance(
                    torch.masked_select(feature_raw_before_normalize, cluster_ids_x).reshape(-1, feature_size)[:used_num].cuda(),
                    torch.masked_select(feature_raw_before_normalize, cluster_ids_x2).reshape(-1, feature_size)[:used_num].cuda()
                )
                
                # 计算后门与干净特征之间的距离
                dis_GTbackdoor2clean = ot.sliced_wasserstein_distance(
                    feature_backdoor_before_normalize_list[0], feature_raw_before_normalize
                )
                
                # 基于聚类的分布距离
                dis_GTbackdoor2clean_cluster_based = dis_GTbackdoor2clean / distance_base
                
                log_info(f"基准距离: {distance_base:.6f}", args)
                log_info(f"基于聚类的分布距离: {dis_GTbackdoor2clean_cluster_based:.6f}", args)
        
        # 计算损失0: 后门特征与参考特征的相似度
        for _index in range(len(feature_reference_list)):
            loss_0_list.append(torch.sum(feature_backdoor_list[_index] * feature_reference_list[_index], dim=-1).unsqueeze(0))
            loss_0_list_cal.append(torch.sum(feature_backdoor_list[_index] * feature_reference_list[_index], dim=-1).mean())
            # 损失1: 增强参考与原始参考的不相似度
            loss_1_list.append(-torch.sum(feature_reference_aug_list[_index] * clean_feature_reference_list[_index], dim=-1).mean())
        
        # 特征相似度分析
        loss_0_list_tensor = torch.cat(loss_0_list, 0)
        std_refs = loss_0_list_tensor.mean(-1).std()
        loss_0_list_tensor_min, index = torch.max(loss_0_list_tensor, dim=0)
        
        # 对每个参考计算选择该参考的样本
        to_ref_list = []
        for _index in range(len(loss_0_list_tensor)):
            to_ref_list.append(torch.argwhere(index == _index).squeeze().tolist())
        
        # 根据模式选择损失0计算方法
        if args.mode == "badencoder":
            loss_0 = -sum(loss_0_list_cal) / len(loss_0_list_cal)
        else:
            loss_0 = -loss_0_list_tensor_min.mean()
        
        # 计算后门与干净特征之间的Wasserstein距离
        dis_GTbackdoor2clean = ot.sliced_wasserstein_distance(
            feature_backdoor_before_normalize_list[0], feature_raw_before_normalize
        )
        total_b2c = dis_GTbackdoor2clean
        
        # 记录Wasserstein距离
        wasserstein_distances.update(dis_GTbackdoor2clean.item())
        
        # 计算后门编码器处理干净数据的特征标准差
        backdoored_clean_dev = feature_raw_before_normalize.std(0).mean()
        
        # 记录分布距离和相似度
        loss_b2c_list.append(total_b2c)
        
        # 计算后门特征内部相似度
        sim_matrix = torch.mm(feature_backdoor_list[0], feature_backdoor_list[0].T)
        distance = (sim_matrix - torch.diag_embed(sim_matrix.diag())).mean()
        sim_backdoor2backdoor_list.append(distance)
        
        # 计算干净特征内部相似度
        sim_matrix = torch.mm(feature_raw, feature_raw.T)
        distance = (sim_matrix - torch.diag_embed(sim_matrix.diag())).mean()
        sim_clean2clean_list.append(distance)
        
        # 损失2: 保持干净图像的特征不变
        loss_2 = -torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()
        
        # 汇总损失
        loss_1 = sum(loss_1_list) / len(loss_1_list)
        cur_sim_backdoor2backdoor = sum(sim_backdoor2backdoor_list) / len(sim_backdoor2backdoor_list)
        cur_sim_clean2clean = sum(sim_clean2clean_list) / len(sim_clean2clean_list)
        
        # 损失3: 参考特征间的相似度
        loss_3_list = []
        for _index in range(len(feature_reference_list)):
            for _index_2 in range(_index + 1, len(feature_reference_list)):
                loss_3_list.append(
                    torch.sum(
                        feature_reference_list[_index] * feature_reference_list[_index_2],
                        dim=-1
                    ).mean()
                )
        
        loss_3 = sum(loss_3_list) / len(loss_3_list) if loss_3_list else torch.tensor(0.0).cuda()
        
        # 分布距离损失
        loss_b2c = sum(loss_b2c_list) / len(loss_b2c_list)
        
        # DRUPE的总损失计算，根据不同阶段和模式采用不同策略
        if args.mode == "drupe":
            if warm_up:
                # 预热阶段
                loss = args.lambda1 * loss_1 + args.lambda2 * loss_2 + 0.5 * loss_3
                if loss_3 < 0.2:
                    loss = loss - 0.2 * loss_3
            else:
                # 正式训练阶段
                if args.encoder_usage_info == "imagenet":
                    stage_1_epoch = 3
                else:
                    stage_1_epoch = 5

                if epoch < stage_1_epoch:
                    # 第一阶段：基础训练+参考多样性
                    loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2 + 2 * std_refs
                    if loss_3 > 0.5:
                        loss = loss + 1 * loss_3
                    elif loss_3 > 0.4:
                        loss = loss + 0.2 * loss_3
                else:
                    # 第二阶段：添加分布对齐和相似度正则化
                    loss = (loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2 + 
                           cost * (std_refs + cur_sim_backdoor2backdoor) + 
                           cost_1 * (loss_b2c / backdoored_clean_dev) + 
                           0.5 * std_refs)
                    
                    # 根据参考多样性动态调整损失
                    if std_refs > 0.1:
                        loss = loss + 1.5 * std_refs
                    
                    # 根据参考相似度动态调整损失
                    if loss_3 > 0.5:
                        loss = loss + 1 * loss_3
                    elif loss_3 > 0.4:
                        loss = loss + 0.2 * loss_3
        
        elif args.mode == "wb":
            # WB模式：加入分布距离
            loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2 + cost * loss_b2c
        
        elif args.mode == "badencoder":
            # 原始BadEncoder模式
            loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2
        
        else:
            raise ValueError(f"无效的模式: {args.mode}")

        # 反向传播
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        
        # 更新度量
        losses.update(loss.item())
        losses_0.update(loss_0.item())
        losses_1.update(loss_1.item())
        losses_2.update(loss_2.item())
        losses_3.update(loss_3.item())
        losses_b2c.update(loss_b2c.item())
        losses_b2c_d_std.update((loss_b2c / backdoored_clean_dev).item())
        sim_backdoor2backdoor.update(cur_sim_backdoor2backdoor.item())
        sim_clean2clean.update(cur_sim_clean2clean.item())
        
        # 定期显示进度
        if i % args.print_freq == 0:
            progress.display(i)
            
            # 记录到日志
            if hasattr(args, 'logger_file'):
                args.logger_file.write(
                    f"E:[{epoch}/{args.epochs}][{i}/{len(data_loader)}],lr:{train_optimizer.param_groups[0]['lr']:.4f},"
                    f"Sb2b:{sim_backdoor2backdoor.avg:.4f},Sc2c:{sim_clean2clean.avg:.4f},"
                    f"l:{losses.avg:.4f},l0:{losses_0.avg:.4f},l1:{losses_1.avg:.4f},"
                    f"l2:{losses_2.avg:.4f},l3:{losses_3.avg:.4f},b2c:{losses_b2c.avg:.4f},"
                    f"b2c/std:{losses_b2c_d_std.avg:.4f},WD:{wasserstein_distances.avg:.6f}"
                    f",JSD:{js_divergences.avg:.6f},JSD_dims:{js_dims_calculated.avg:.0f}\n"
                )
                args.logger_file.flush()
    
    print("开始计算线性可分性")
    # 计算线性可分性（如果收集了特征）
    linear_separability = 0.0
    if hasattr(args, 'collect_features') and args.collect_features:
        # 合并收集的特征
        shadow_features = torch.cat(shadow_features_list, dim=0)
        target_features = torch.cat(target_features_list, dim=0)
        
        # 计算线性可分性
        linear_separability = compute_linear_separability(shadow_features, target_features)
        log_info(f"Epoch {epoch}: 线性可分性 = {linear_separability:.4f}", args)
    
    # Epoch结束，计算特征分布差异 (JS散度)
    # ------------------ 已禁用 JS 散度计算 ------------------
    avg_js_dist = 0.0
    num_js_dims = 0
    # if len(all_feature_raw_tensors) > 0 and len(all_feature_backdoor_tensors) > 0:
    #     # 合并收集到的特征
    #     all_feature_raw_np = torch.cat(all_feature_raw_tensors, dim=0).numpy()
    #     all_feature_backdoor_np = torch.cat(all_feature_backdoor_tensors, dim=0).numpy()
    #     
    #     # 调用新函数计算JS散度
    #     max_kde_dims = getattr(args, 'max_kde_dims', 1000) # 默认为1000
    #     js_seed = getattr(args, 'kde_seed', epoch) # 使用epoch作为种子，或从args获取

    #     avg_js_dist, num_js_dims = calculate_js_divergence_per_dim(
    #         all_feature_raw_np,
    #         all_feature_backdoor_np,
    #         max_dims_to_sample=max_kde_dims,
    #         random_seed_for_sampling=js_seed
    #     )
    #     
    #     js_divergences.update(avg_js_dist, n=1) # n=1 because it's an epoch-level average
    #     js_dims_calculated.update(num_js_dims, n=1)

    #     log_info(f"Epoch {epoch}: 特征分布Jensen-Shannon散度 = {avg_js_dist:.6f} (calculated over {num_js_dims} dimensions)", args)
    #     if hasattr(args, 'logger_file'):
    #         args.logger_file.write(f"Epoch {epoch}: 特征分布Jensen-Shannon散度 = {avg_js_dist:.6f} (calculated over {num_js_dims} dims)\n")
    #         args.logger_file.flush()
    # -------------------------------------------------------

    # 自适应调整代价系数
    if not warm_up and epoch > args.fix_epoch:
        # 初始化阈值
        if args.encoder_usage_info in ["imagenet"]:
            l0_threshold = -0.91
        else:
            l0_threshold = -0.96
        
        # 根据损失情况调整cost系数
        if (losses_0.avg < l0_threshold and losses_1.avg < -0.9 and losses_2.avg < -0.9):
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1

        if cost_up_counter >= patience:
            cost_up_counter = 0
            if cost == 0:
                cost = init_cost
            else:
                cost *= cost_multiplier_up
        elif cost_down_counter >= patience:
            cost_down_counter = 0
            cost /= cost_multiplier_down
        
        # 设置相似度阈值
        if args.encoder_usage_info in ["CLIP"]:
            b2b_sim_threshold = 0.8
        else:
            b2b_sim_threshold = 0.6
            
        # 根据相似度调整cost_1系数
        if (losses_0.avg < l0_threshold and 
            losses_1.avg < -0.9 and 
            losses_2.avg < -0.9 and 
            sim_backdoor2backdoor.avg < b2b_sim_threshold):
            
            cost_up_counter_1 += 1
            cost_down_counter_1 = 0

            if sim_backdoor2backdoor.avg < (b2b_sim_threshold - 0.1):
                cost_up_counter -= 1

            args.measure = loss_b2c.avg
        else:
            cost_up_counter_1 = 0
            cost_down_counter_1 += 1

        if cost_up_counter_1 >= patience:
            cost_up_counter_1 = 0
            if cost_1 == 0:
                cost_1 = init_cost_1
            else:
                cost_1 *= cost_multiplier_up
        elif cost_down_counter_1 >= patience:
            cost_down_counter_1 = 0
            cost_1 /= cost_multiplier_down
    
    return losses.avg, wasserstein_distances.avg, linear_separability, js_divergences.avg, js_dims_calculated.avg


def run_drupe(args, pretrained_encoder, shadow_dataset=None, memory_dataset=None, 
             test_data_clean=None, test_data_backdoor=None, downstream_train_dataset=None):
    """
    运行DRUPE后门攻击
    
    Args:
        args: 配置参数
        pretrained_encoder: 预训练编码器模型
        shadow_dataset: 用于注入后门的数据集
        memory_dataset: 内存数据集，用于评估
        test_data_clean: 干净测试数据集
        test_data_backdoor: 有毒测试数据集
        downstream_train_dataset: 下游训练数据集
    
    Returns:
        训练好的后门编码器和评估结果
    """
    start_time = time.time()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据加载器
    train_loader = DataLoader(
        shadow_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    # 复制出后门初始化模型
    backdoored_model = copy.deepcopy(pretrained_encoder)
    
    # 创建优化器
    # 过去的代码，暂时注释
    # if args.encoder_usage_info in ['cifar10', 'stl10']:
    #     optimizer = torch.optim.SGD(backdoored_model.f.parameters(), lr=args.lr, 
    #                                weight_decay=args.weight_decay, momentum=args.momentum)
    # elif args.encoder_usage_info in ['imagenet', 'CLIP']:
    #     optimizer = torch.optim.SGD(backdoored_model.visual.parameters(), lr=args.lr, 
    #                                weight_decay=args.weight_decay, momentum=args.momentum)
    # else:
    #     raise NotImplementedError(f"未支持的编码器使用信息: {args.encoder_usage_info}")
    assert not hasattr(backdoored_model, 'visual'), "backdoored_model 不应该有 visual 属性，我暂时还没实现"
    optimizer = torch.optim.SGD(backdoored_model.parameters(), lr=args.lr, 
                               weight_decay=args.weight_decay, momentum=args.momentum)
    

    
    # 创建检查点目录
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化DRUPE的代价系数和计数器
    global patience, cost_multiplier_up, cost_multiplier_down, init_cost, cost, cost_up_counter, cost_down_counter
    global init_cost_1, cost_1, cost_up_counter_1, cost_down_counter_1
    
    patience = 1
    cost_multiplier_up = 1.25
    cost_multiplier_down = 1.25 ** 1.25
    
    # 根据编码器调整初始代价系数
    if args.encoder_usage_info in ["CLIP"]:
        init_cost = 0.01
        init_cost_1 = 0.0001
    else:
        init_cost = 0.1
        init_cost_1 = 0.001
    
    cost = 0  # 初始为0，自适应调整
    cost_up_counter = 0
    cost_down_counter = 0
    
    cost_1 = 0  # 初始为0，自适应调整
    cost_up_counter_1 = 0
    cost_down_counter_1 = 0
    
    args.measure = 0
    measure_best = float('inf')
    
    # 添加特征收集标志
    args.collect_features = True
    
    # 初始化指标记录器
    metric_logger = MetricLogger()

    # ====== 初始下游任务评估 ======
    log_info("\n====== 初始下游任务评估 ======", args)

    if all(x is not None for x in [downstream_train_dataset, test_data_clean, test_data_backdoor]):
        try:
            init_results = train_downstream_classifier(
                args, backdoored_model, downstream_train_dataset,
                test_data_clean, test_data_backdoor
            )
            log_info(f"初始 BA={init_results['BA']:.2f}%, ASR={init_results['ASR']:.2f}%", args)
        except Exception as e:
            log_info(f"初始下游任务评估出错: {e}", args)
    else:
        log_info("缺少下游评估所需的数据集，跳过初始评估。", args)
    
    # 训练循环
    for epoch in range(args.epochs):
        log_info("=================================================", args)
        
        # 确定当前训练阶段的参数
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            warm_up = (epoch < args.warm_up_epochs)
            get_clean_dev = (epoch == 0)
            cal_cluster_based_dist = (epoch % 10 == 0)
            
            train_loss, wasserstein_distance, linear_separability, js_divergence, num_js_dims = train_drupe(
                backdoored_model.f, pretrained_encoder.f, train_loader, 
                optimizer, epoch, args, warm_up, get_clean_dev, cal_cluster_based_dist
            )
            
        elif args.encoder_usage_info in ['imagenet', 'CLIP']:
            # 根据编码器类型确定预热轮数
            if args.encoder_usage_info == 'imagenet':
                warm_up_epoch = 2
            else:  # CLIP
                warm_up_epoch = 1
                
            warm_up = (epoch < warm_up_epoch)
            get_clean_dev = (epoch == 0)
            cal_cluster_based_dist = (epoch % 10 == 0)
            
            assert not hasattr(backdoored_model, 'visual'), "backdoored_model 不应该有 visual 属性，我暂时还没实现"
            train_loss, wasserstein_distance, linear_separability, js_divergence, num_js_dims = train_drupe(
                backdoored_model, pretrained_encoder, train_loader, 
                optimizer, epoch, args, warm_up, get_clean_dev, cal_cluster_based_dist
            )
            
        else:
            raise NotImplementedError(f"未支持的编码器使用信息: {args.encoder_usage_info}")
        
        # 记录当前epoch的指标
        metric_logger.log_epoch_metrics(epoch, wasserstein_distance, linear_separability, js_divergence, num_js_dims)
        
        # 根据分布度量保存最佳模型
        log_info(f"当前度量: {args.measure}, 最佳度量: {measure_best}", args)
        if epoch > 24 and args.measure < measure_best:
            measure_best = args.measure
            best_checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': backdoored_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, best_checkpoint_path)
            log_info(f"保存最佳模型到: {best_checkpoint_path}, 度量: {measure_best:.6f}", args)
        
        # 定期保存检查点
        if (epoch+1) % args.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': backdoored_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)
            log_info(f"保存检查点到: {checkpoint_path}", args)

            # 定期评估下游任务
            log_info("\n====== 阶段性下游任务评估 ======", args)
            if all(x is not None for x in [downstream_train_dataset, test_data_clean, test_data_backdoor]):
                try:
                    # 使用导入的 train_downstream_classifier
                    model_results = train_downstream_classifier(
                        args, backdoored_model, downstream_train_dataset,
                        test_data_clean, test_data_backdoor
                    )
                    log_info(f"阶段 {epoch}/{args.epochs} 下游任务评估完成: BA={model_results['BA']:.2f}%, ASR={model_results['ASR']:.2f}%", args)
                except Exception as e:
                    log_info(f"下游任务评估出错: {e}", args)
            else:
                log_info("缺少下游评估所需的数据集，跳过评估。", args)
    
    # 加载最佳模型进行最终评估
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        backdoored_model.load_state_dict(checkpoint['state_dict'])
        log_info("已加载最佳模型进行最终评估", args)
    else:
        log_info("未找到最佳模型文件，将使用最后训练的模型进行评估", args)

    
    # ====== 最终下游任务评估 ======
    final_results = None
    log_info("\n====== 最终下游任务评估 ======", args)

    if all(x is not None for x in [downstream_train_dataset, test_data_clean, test_data_backdoor]):
        try:
            # 使用导入的 train_downstream_classifier
            final_results = train_downstream_classifier(
                args, backdoored_model, downstream_train_dataset, 
                test_data_clean, test_data_backdoor
            )
            log_info(f"最终结果: BA={final_results['BA']:.2f}%, ASR={final_results['ASR']:.2f}%", args)
        except Exception as e:
            log_info(f"最终下游任务评估出错: {e}", args)
    else:
        log_info("缺少下游评估所需的数据集，无法进行最终评估。", args)

    
    elapsed_time = time.time() - start_time
    log_info(f"DRUPE训练完成，耗时: {elapsed_time:.2f}秒", args)
    
    return backdoored_model, final_results 