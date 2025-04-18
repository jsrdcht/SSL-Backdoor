#!/usr/bin/env python
# 实现分布式模型评估的接口

# 标准库导入
import os
import time
import argparse
from typing import Dict, Any, Tuple, List, Optional, Callable

# 第三方库导入
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.models as models

# 项目内部导入
from tools.eval_utils import AverageMeter, ProgressMeter, accuracy, save_checkpoint
from ssl_backdoor.datasets.dataset import FileListDataset, OnlineUniversalPoisonedValDataset
from ssl_backdoor.utils.utils import interpolate_pos_embed, get_channels
import ssl_backdoor.ssl_trainers.models_vit as models_vit


class Normalize(nn.Module):
    """特征归一化层，将特征向量标准化为单位长度"""
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class FullBatchNorm(nn.Module):
    """使用预计算的统计量的批归一化层
    
    Args:
        var: 预计算的特征方差
        mean: 预计算的特征均值
    """
    def __init__(self, var, mean):
        super().__init__()
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


def load_model_weights(model, wts_path: str) -> Dict[str, Any]:
    """
    加载模型权重
    
    Args:
        model: 待加载权重的模型
        wts_path: 权重文件路径
        
    Returns:
        模型状态字典
        
    Raises:
        ValueError: 如果找不到有效的权重
    """
    checkpoint = torch.load(wts_path, map_location='cpu')
    
    # 按优先级顺序尝试不同的键名
    for key in ['model', 'state_dict', 'model_state_dict']:
        if key in checkpoint:
            return checkpoint[key]
            
    raise ValueError(f'无法在 {wts_path} 中找到模型权重')


def get_backbone_model(arch, wts_path, device, dataset='imagenet100'):
    """获取并加载预训练的主干网络。"""
    from ssl_backdoor.utils.model_utils import get_backbone_model
    
    return get_backbone_model(arch, wts_path, device, dataset)


def get_transforms(dataset_name):
    """
    获取数据集对应的转换
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        (train_transform, val_transform): 训练和验证数据转换
        
    Raises:
        ValueError: 如果数据集不受支持
    """
    # 数据集归一化参数
    dataset_params = {
        'imagenet100': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'train_size': 224,
            'val_size': 224,
            'resize': 256,
        },
        'cifar10': {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010],
            'train_size': 32,
            'val_size': 32,
            'resize': None,  # 不需要额外的resize
        },
        'stl10': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'train_size': 96,
            'val_size': 96,
            'resize': None,  # 直接resize到目标大小
        }
    }
    
    if dataset_name not in dataset_params:
        raise ValueError(f"未知数据集 '{dataset_name}'")
    
    params = dataset_params[dataset_name]
    normalize = transforms.Normalize(mean=params['mean'], std=params['std'])
    
    # 创建训练转换
    train_transforms = [
        transforms.RandomCrop(params['train_size'], padding=4) if dataset_name == 'cifar10' else transforms.RandomResizedCrop(params['train_size'], scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    print("train_transforms", train_transforms)
    
    # 创建验证转换 - 确保所有图像大小一致
    val_transforms = [
        # 固定大小的调整，强制所有图像具有相同的形状
        transforms.Resize((params['val_size'], params['val_size'])),
    ]
    
    val_transforms.extend([
        transforms.ToTensor(),
        normalize,
    ])
    
    print("val_transforms", val_transforms)
    
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


def get_dataloaders(args, val_transform):
    """
    创建用于评估的数据加载器
    
    Args:
        args: 参数对象，包含数据加载配置
        val_transform: 验证数据转换
        
    Returns:
        tuple: (train_val_loader, val_loader, val_poisoned_loader)
    """
    # 创建数据集
    train_dataset = FileListDataset(args, args.train_file, val_transform)
    val_dataset = FileListDataset(args, args.val_file, val_transform)
    val_poisoned_dataset = OnlineUniversalPoisonedValDataset(args, args.val_file, val_transform)
    
    # 通用数据加载器参数
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': min(4, args.workers),  # 限制worker数量
        'pin_memory': True,
        'drop_last': False,
    }
    
    # 分布式训练配置
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank)
        val_poisoned_sampler = torch.utils.data.distributed.DistributedSampler(
            val_poisoned_dataset, num_replicas=args.world_size, rank=args.rank)
        
        # 更新loader参数
        train_kwargs = {**loader_kwargs, 'sampler': train_sampler, 'shuffle': False}
        val_kwargs = {**loader_kwargs, 'sampler': val_sampler, 'shuffle': False}
        val_poisoned_kwargs = {**loader_kwargs, 'sampler': val_poisoned_sampler, 'shuffle': False}
    else:
        # 非分布式训练
        train_kwargs = {**loader_kwargs, 'shuffle': True}
        val_kwargs = {**loader_kwargs, 'shuffle': False}
        val_poisoned_kwargs = {**loader_kwargs, 'shuffle': False}
    
    # 创建数据加载器
    train_val_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)
    val_poisoned_loader = torch.utils.data.DataLoader(val_poisoned_dataset, **val_poisoned_kwargs)

    return train_val_loader, val_loader, val_poisoned_loader


def get_feats(loader, model):
    """从主干网络提取特征。"""
    model.eval()
    
    # 特征和标签的存储
    feats, labels = None, None
    
    # 如果是分布式训练，先确保所有进程同步
    is_distributed = dist.is_initialized()
    world_size = dist.get_world_size() if is_distributed else 1
    rank = dist.get_rank() if is_distributed else 0
    
    # 设置进度条
    progress = ProgressMeter(
        len(loader),
        [AverageMeter('Time', ':6.3f')],
        prefix='提取特征: ')

    with torch.no_grad():
        end = time.time()
        # 使用指针来跟踪处理的样本数量
        ptr = 0
        
        for i, (images, target) in enumerate(loader):
            # 测量时间
            progress.meters[0].update(time.time() - end)
            end = time.time()
            
            images = images.cuda(non_blocking=True).contiguous()
            cur_targets = target.cpu()
            # 将特征归一化
            cur_feats = F.normalize(model(images), dim=1).cpu()
            
            # 获取当前批次的特征维度和索引
            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr
            
            # 初始化特征和标签张量 (仅在第一次迭代时)
            if ptr == 0:
                # 计算当前进程需要处理的样本数量
                total_size = len(loader.dataset)
                if is_distributed:
                    # 在分布式环境中，每个进程只处理数据集的一部分
                    samples_per_rank = total_size // world_size + (1 if total_size % world_size > rank else 0)
                else:
                    samples_per_rank = total_size
                
                feats = torch.zeros((samples_per_rank, D)).float()
                labels = torch.zeros(samples_per_rank).long()
            
            # 使用index_copy_安全地存储特征和标签
            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            ptr += B
            
            # 显示进度
            if i % 10 == 0 and (rank == 0):
                progress.display(i)

        # 截断为实际处理的样本数量 (如果feats和labels预分配的空间大于实际需要)
        if ptr < feats.shape[0]:
            feats = feats[:ptr]
            labels = labels[:ptr]
        
        # 如果是分布式训练，收集所有进程的特征和标签
        if is_distributed:
            # 同步所有进程，确保所有进程都完成了特征提取
            dist.barrier()
            
            # 创建用于收集所有进程特征和标签的列表
            all_feats = [None for _ in range(world_size)]
            all_labels = [None for _ in range(world_size)]
            
            # 收集每个进程的特征维度
            local_feats_shape = torch.tensor([feats.shape[0], feats.shape[1]], dtype=torch.long).cuda()
            all_shapes = [torch.zeros(2, dtype=torch.long).cuda() for _ in range(world_size)]
            dist.all_gather(all_shapes, local_feats_shape)
            
            # 等待所有进程
            dist.barrier()
            
            # 在rank 0上整合所有特征
            if rank == 0:
                total_samples = sum(shape[0].item() for shape in all_shapes)
                print(f"总样本数: {total_samples}, 特征维度: {feats.shape[1]}")
                
                # 创建全局特征和标签张量
                global_feats = torch.zeros((total_samples, feats.shape[1])).float()
                global_labels = torch.zeros(total_samples).long()
                
                # 复制本地特征和标签
                global_feats[:feats.shape[0]] = feats
                global_labels[:feats.shape[0]] = labels
                
                # 收集其他进程的特征和标签
                start_idx = feats.shape[0]
                for i in range(1, world_size):
                    num_samples = all_shapes[i][0].item()
                    if num_samples > 0:
                        # 创建临时存储
                        temp_feats = torch.zeros((num_samples, feats.shape[1])).float().cuda()
                        temp_labels = torch.zeros(num_samples).long().cuda()
                        
                        # 接收其他进程的数据
                        dist.recv(temp_feats, src=i)
                        dist.recv(temp_labels, src=i)
                        
                        # 存储到全局张量
                        global_feats[start_idx:start_idx+num_samples] = temp_feats.cpu()
                        global_labels[start_idx:start_idx+num_samples] = temp_labels.cpu()
                        start_idx += num_samples
                
                feats = global_feats
                labels = global_labels
            else:
                # 其他进程发送数据到rank 0
                if feats.shape[0] > 0:
                    dist.send(feats.cuda(), dst=0)
                    dist.send(labels.cuda(), dst=0)
            
            # 再次同步所有进程
            dist.barrier()
            
            # 广播最终的特征和标签从rank 0到所有进程
            if rank == 0:
                # 广播特征维度
                feat_shape = torch.tensor([feats.shape[0], feats.shape[1]], dtype=torch.long).cuda()
            else:
                feat_shape = torch.zeros(2, dtype=torch.long).cuda()
            
            dist.broadcast(feat_shape, src=0)
            
            if rank != 0:
                # 其他进程根据广播的维度创建张量
                feats = torch.zeros((feat_shape[0].item(), feat_shape[1].item())).float().cuda()
                labels = torch.zeros(feat_shape[0].item()).long().cuda()
            else:
                feats = feats.cuda()
                labels = labels.cuda()
            
            # 广播特征和标签
            dist.broadcast(feats, src=0)
            dist.broadcast(labels, src=0)
            
            # 转回CPU
            feats = feats.cpu()
            labels = labels.cpu()

    return feats, labels


def train_linear_classifier(train_loader, backbone, linear, optimizer, epoch, args):
    """训练线性分类器。"""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]")

    # 切换到训练模式
    backbone.eval()
    linear.train()
    
    # 如果是分布式训练，设置sampler的epoch
    if args.distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # 测量数据加载时间
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # 计算输出
        with torch.no_grad():
            output = backbone(images)
        output = linear(output)
        loss = F.cross_entropy(output, target)

        # 测量准确率并记录损失
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        # 在分布式训练中，需要平均所有GPU上的损失和准确率
        if args.distributed:
            # 收集所有GPU上的损失和准确率
            loss_list = [torch.zeros_like(loss) for _ in range(args.world_size)]
            acc1_list = [torch.zeros_like(acc1) for _ in range(args.world_size)]
            acc5_list = [torch.zeros_like(acc5) for _ in range(args.world_size)]
            
            # 收集损失
            dist.all_gather(loss_list, loss.detach())
            loss_mean = torch.mean(torch.stack(loss_list))
            
            # 收集准确率
            dist.all_gather(acc1_list, acc1.detach())
            dist.all_gather(acc5_list, acc5.detach())
            acc1_mean = torch.mean(torch.stack(acc1_list))
            acc5_mean = torch.mean(torch.stack(acc5_list))
            
            # 更新指标
            losses.update(loss_mean.item(), images.size(0) * args.world_size)
            top1.update(acc1_mean.item(), images.size(0) * args.world_size)
            top5.update(acc5_mean.item(), images.size(0) * args.world_size)
        else:
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        # 计算梯度并执行SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 测量时间
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        if i % args.print_freq == 0 and (args.rank == 0 or not args.distributed):
            # 直接获取进度消息
            progress_msg = progress.display(i)
            
            # 始终打印到控制台
            print(f"训练进度: {progress_msg}", flush=True)

    # 如果是分布式训练，确保所有进程同步
    if args.distributed:
        dist.barrier()
    
    return top1.avg


def validate(val_loader, backbone, linear, args):
    """验证分类器性能。"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # 切换到评估模式
    backbone.eval()
    linear.eval()
    
    # 如果是分布式训练，设置sampler的epoch
    if args.distributed and hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'set_epoch'):
        val_loader.sampler.set_epoch(0)  # 设置为固定值，确保每次验证使用相同的样本顺序

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # 计算输出
            output = backbone(images)
            output = linear(output)
            loss = F.cross_entropy(output, target)

            # 测量准确率并记录损失
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            # 在分布式训练中，需要平均所有GPU上的损失和准确率
            if args.distributed:
                # 收集所有GPU上的损失和准确率
                loss_list = [torch.zeros_like(loss) for _ in range(args.world_size)]
                acc1_list = [torch.zeros_like(acc1) for _ in range(args.world_size)]
                acc5_list = [torch.zeros_like(acc5) for _ in range(args.world_size)]
                
                # 收集损失
                dist.all_gather(loss_list, loss.detach())
                loss_mean = torch.mean(torch.stack(loss_list))
                
                # 收集准确率
                dist.all_gather(acc1_list, acc1.detach())
                dist.all_gather(acc5_list, acc5.detach())
                acc1_mean = torch.mean(torch.stack(acc1_list))
                acc5_mean = torch.mean(torch.stack(acc5_list))
                
                # 更新指标
                losses.update(loss_mean.item(), images.size(0) * args.world_size)
                top1.update(acc1_mean.item(), images.size(0) * args.world_size)
                top5.update(acc5_mean.item(), images.size(0) * args.world_size)
            else:
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

            # 测量时间
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and (args.rank == 0 or not args.distributed):
                # 直接获取进度消息
                progress_msg = progress.display(i)
                print(f"验证进度: {progress_msg}", flush=True)


        if args.rank == 0 or not args.distributed:
            result_message = ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
            print(result_message)
    
    # 如果是分布式训练，确保所有进程同步
    if args.distributed:
        dist.barrier()

    return top1.avg


def validate_with_conf_matrix(val_loader, backbone, linear, args):
    """验证并生成混淆矩阵。"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # 切换到评估模式
    backbone.eval()
    linear.eval()

    num_classes = 100 if args.dataset == 'imagenet100' else 10
    conf_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # 计算输出
            output = backbone(images)
            output = linear(output)
            loss = F.cross_entropy(output, target)

            # 测量准确率并记录损失
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # 更新混淆矩阵
            _, pred = output.topk(1, 1, True, True)
            pred_numpy = pred.cpu().numpy()
            target_numpy = target.cpu().numpy()
            
            for elem in range(target.size(0)):
                conf_matrix[target_numpy[elem], int(pred_numpy[elem])] += 1

            # 测量时间
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and (args.rank == 0 or not args.distributed):
                # 直接获取进度消息
                progress_msg = progress.display(i)
                print(f"混淆矩阵验证进度: {progress_msg}", flush=True)

        if args.rank == 0 or not args.distributed:
            result_message = ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
            print(result_message)

    return top1.avg, top5.avg, conf_matrix


def test_model(model_path, epoch, args, logger=None, config=None):
    """
    直接测试模型，不使用嵌套函数
    
    Args:
        model_path: 模型权重路径
        epoch: 当前epoch
        args: 训练参数
        logger: 日志记录器，从训练代码传入，确保日志一致性
        config: 额外的配置参数，如果为None则从args中获取
        
    Returns:
        clean_acc: 干净验证集准确率
        poison_acc: 中毒验证集准确率
        asr: 攻击成功率
    """
    
    # 分布式环境中同步模型路径
    if args.distributed:
        if not os.path.exists(model_path) and args.rank != 0:
            print(f"Rank {args.rank} 等待模型路径广播...")
            object_list = [None]
            dist.broadcast_object_list(object_list, src=0)
            if object_list[0] is not None:
                model_path = object_list[0]
                print(f"Rank {args.rank} 收到模型路径: {model_path}")
            else:
                raise ValueError(f"Rank {args.rank} 未收到模型路径广播")
        
        # 确保所有进程同步
        dist.barrier()
        
    # 解析参数
    def _get_param(name, default=None):
        # 按优先级顺序：1) test_前缀参数 2) 直接参数 3) config中的参数 4) 默认值
        test_name = f"test_{name}"
        if hasattr(args, test_name):
            return getattr(args, test_name)
        elif hasattr(args, name):
            return getattr(args, name)
        elif config and name in config:
            return config[name]
        return default
    
    # 初始化评估参数
    eval_args = {
        # 必要参数
        'train_file': _get_param('train_file'),
        'val_file': _get_param('val_file'),
        'dataset': _get_param('dataset'),
        'attack_target': _get_param('attack_target'),
        'trigger_path': _get_param('trigger_path'),
        'trigger_size': _get_param('trigger_size'),
        'trigger_insert': _get_param('trigger_insert'),
        'attack_algorithm': _get_param('attack_algorithm'),
        
        # 可选参数
        'workers': _get_param('workers', 4),
        'batch_size': _get_param('batch_size', 64),
        'print_freq': _get_param('print_freq', 10),
        'distributed': _get_param('distributed', False),
        'rank': _get_param('rank', 0),
        'world_size': _get_param('world_size', 1),
        
        # 固定参数
        'weights': model_path,
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'epochs': 40,
        'lr_schedule': '15,30,40',
        'arch': args.arch
    }
    
    # 验证必要参数
    required_params = ['train_file', 'val_file', 'dataset', 'attack_target', 
                      'trigger_path', 'trigger_insert', 'attack_algorithm']
    missing = [p for p in required_params if eval_args[p] is None]
    if missing:
        raise ValueError(f"缺少必要参数: {', '.join(missing)}")
    
    # 打印参数信息（仅主进程）
    if eval_args['rank'] == 0:
        print("测试参数配置:")
        for key in required_params + ['batch_size', 'epochs']:
            print(f"  - {key}: {eval_args[key]}")
    
    # 创建参数对象
    args_obj = argparse.Namespace(**eval_args)
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.rank}' if args.distributed else 'cuda')
    else:
        device = torch.device('cpu')
    
    # 获取数据转换
    train_transform, val_transform = get_transforms(args_obj.dataset)
    
    # 同步所有进程，确保数据加载器创建前所有进程都准备好
    if args.distributed:
        dist.barrier()
    
    # 获取数据加载器
    train_val_loader, val_loader, val_poisoned_loader = get_dataloaders(args_obj, val_transform)
    
    # 加载主干网络
    backbone = get_backbone_model(args_obj.arch, model_path, device, args_obj.dataset)
    # 不要对没有需要梯度的backbone使用DDP
    if args.distributed:
        # 检查模型是否有需要梯度的参数
        has_grad_params = any(p.requires_grad for p in backbone.parameters())
        if has_grad_params:
            backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[args.rank], find_unused_parameters=True)
        else:
            print(f"注意: backbone没有需要梯度的参数，不使用DistributedDataParallel包装 (rank {args.rank})")
    
    # 提取特征统计
    print(f"get_feats ing, the rank is {args.rank}")
    train_feats, _ = get_feats(train_val_loader, backbone)
    print(f"get_feats done, the rank is {args.rank}")
    
    # 计算训练特征的方差和均值
    train_var, train_mean = torch.var_mean(train_feats, dim=0)
    
    # 创建线性分类器
    arch = args_obj.arch if 'moco_' not in args_obj.arch else args_obj.arch.replace('moco_', '')
    nb_classes = 100 if args_obj.dataset == 'imagenet100' else 10

    
    linear = nn.Sequential(
        Normalize(),
        FullBatchNorm(train_var, train_mean),
        nn.Linear(get_channels(arch), nb_classes),
    ).to(device)
    
    # 同步所有进程，确保所有进程都创建了线性分类器
    if args.distributed:
        dist.barrier()
        print(f"同步所有进程，确保都创建了线性分类器, rank: {args.rank}")
        
    if args.distributed:
        # 检查模型是否有需要梯度的参数
        has_grad_params = any(p.requires_grad for p in linear.parameters())
        if has_grad_params:
            try:            
                linear = torch.nn.parallel.DistributedDataParallel(
                    linear, 
                    device_ids=[args.rank], 
                    find_unused_parameters=True
                )
            except Exception as e:
                print(f"DDP包装失败: {e}，继续使用非DDP模型，rank: {args.rank}")
        else:
            print(f"注意: linear分类器没有需要梯度的参数，不使用DistributedDataParallel包装，rank: {args.rank}")
    
    # 设置优化器和学习率调度器
    optimizer = torch.optim.SGD(linear.parameters(), args_obj.lr,
                                momentum=args_obj.momentum,
                                weight_decay=args_obj.weight_decay)
    
    sched = [int(x) for x in args_obj.lr_schedule.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sched)
    

    # 训练线性分类器
    best_acc1 = 0.0
    best_linear_state = None
    
    for e in range(args_obj.epochs):
        print(f"Linear eval epoch {e+1}/{args_obj.epochs}")
        
        # 训练一个epoch
        train_linear_classifier(train_val_loader, backbone, linear, optimizer, e, args_obj)
        
        # 验证
        acc1 = validate(val_loader, backbone, linear, args_obj)
        
        # 更新学习率
        lr_scheduler.step()
        
        # 只在rank 0保存最佳模型
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_linear_state = linear.state_dict()
    
    # 将最佳模型从rank 0广播到所有进程
    if args.distributed:
        # 同步所有进程
        dist.barrier()
        
        # 如果当前进程是rank 0，则广播最佳线性分类器状态
        if args.rank == 0:
            # 首先创建一个包含状态字典的列表，用于广播
            state_list = [best_linear_state]
        else:
            # 其他进程创建一个空列表，用于接收状态字典
            state_list = [None]
        
        # 广播状态字典
        dist.broadcast_object_list(state_list, src=0)
        
        # 更新非rank 0进程的最佳线性分类器状态
        if args.rank != 0:
            best_linear_state = state_list[0]
        
        # 再次同步所有进程
        dist.barrier()
    
    # 加载最佳模型权重
    linear.load_state_dict(best_linear_state)
    
    # 最终评估
    clean_acc, _, clean_conf_matrix = validate_with_conf_matrix(val_loader, backbone, linear, args_obj)
    poison_acc, _, poison_conf_matrix = validate_with_conf_matrix(val_poisoned_loader, backbone, linear, args_obj)
    
    # 计算攻击成功率 (Attack Success Rate)
    assert args_obj.attack_target is not None, "攻击目标未指定"
    attack_target = args_obj.attack_target
    
    # 在毒化样本中计算攻击成功率
    non_target_total = 0
    non_target_success = 0
    
    for i in range(poison_conf_matrix.shape[0]):
        if i != attack_target:  # 排除本身就是目标类的样本
            class_samples = np.sum(poison_conf_matrix[i, :])
            if class_samples > 0:
                non_target_total += class_samples
                non_target_success += poison_conf_matrix[i, attack_target]
    
    asr = (non_target_success / non_target_total * 100) if non_target_total > 0 else 0.0
    
    if args.rank == 0 or not args.distributed:
        result_message = f"Epoch {epoch} | Clean Acc: {clean_acc:.2f}% | Poisoned Acc: {poison_acc:.2f}% | Attack Success Rate: {asr:.2f}%"
        print(result_message)
        print(f"Clean Confusion Matrix:\n{np.array2string(clean_conf_matrix, precision=0)}")
        print(f"Poison Confusion Matrix:\n{np.array2string(poison_conf_matrix, precision=0)}")

        if logger:
            if hasattr(logger, 'log'):
                # 记录基本指标
                logger.log({
                    "eval/clean_acc": clean_acc,
                    "eval/poison_acc": poison_acc,
                    "eval/attack_success_rate": asr
                }, step=epoch)
                
                # 创建混淆矩阵的可视化并记录
                try:
                    import matplotlib.pyplot as plt
                    import io
                    from PIL import Image
                    
                    # 准备类别名称
                    class_names = [str(i) for i in range(clean_conf_matrix.shape[0])]
                    num_classes = clean_conf_matrix.shape[0]
                    
                    # 根据类别数量动态调整图片大小和DPI
                    fig_size = max(10, num_classes * 0.5)  # 随类别数增加而增加
                    dpi = max(100, min(300, 100 + num_classes * 2))  # 提高DPI但设置上限
                    
                    # 记录干净数据集的混淆矩阵
                    plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
                    plt.imshow(clean_conf_matrix, cmap='Blues')
                    plt.colorbar()
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title(f'Clean Confusion Matrix (Epoch {epoch})')
                    # 如果类别较多，可能需要调整标签字体大小
                    if num_classes > 20:
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                    if num_classes <= 30:  # 类别不太多时显示刻度
                        plt.xticks(range(num_classes), class_names)
                        plt.yticks(range(num_classes), class_names)
                    
                    # 将图像保存到内存缓冲区
                    clean_buf = io.BytesIO()
                    plt.savefig(clean_buf, format='png', bbox_inches='tight')
                    clean_buf.seek(0)
                    
                    # 创建PIL图像对象
                    clean_img = Image.open(clean_buf)
                    plt.close()
                    plt.clf()  # <---- 添加: 显式清除图形状态
                    
                    # 记录毒化数据集的混淆矩阵
                    plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
                    plt.imshow(poison_conf_matrix, cmap='Reds')
                    plt.colorbar()
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title(f'Poison Confusion Matrix (Epoch {epoch})')
                    if num_classes > 20:
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                    if num_classes <= 30:  # 类别不太多时显示刻度
                        plt.xticks(range(num_classes), class_names)
                        plt.yticks(range(num_classes), class_names)
                    
                    # 将图像保存到内存缓冲区
                    poison_buf = io.BytesIO()
                    plt.savefig(poison_buf, format='png', bbox_inches='tight')
                    poison_buf.seek(0)
                    
                    # 创建PIL图像对象
                    poison_img = Image.open(poison_buf)
                    plt.close()
                    
                    # 将PIL图像转为numpy数组以便统一处理
                    clean_np_img = np.array(clean_img)
                    poison_np_img = np.array(poison_img)
                    
                    # 使用Logger类的标准接口添加图像
                    logger.add_image('eval/clean_confusion_matrix', clean_np_img, epoch) # TODO: 目前的图像上传没有进度记录，只有一个最新的图像
                    logger.add_image('eval/poison_confusion_matrix', poison_np_img, epoch)
                    
                    # 对于wandb，额外添加表格格式的混淆矩阵（这是wandb特有的功能）
                    if logger.log_type == 'wandb':
                        import wandb
                        # 使用wandb原生表格格式记录混淆矩阵
                        clean_table = wandb.Table(
                            columns=["True/Pred"] + class_names,
                            data=[[class_names[i]] + [float(x) for x in clean_conf_matrix[i].tolist()] for i in range(len(class_names))]
                        )
                        # <---- 添加: 调试打印
                        print(f"DEBUG [rank {args.rank}]: Data for poison_table before creation:\n{np.array2string(poison_conf_matrix, precision=0)}")
                        poison_table = wandb.Table(
                            columns=["True/Pred"] + class_names,
                            data=[[class_names[i]] + [float(x) for x in poison_conf_matrix[i].tolist()] for i in range(len(class_names))]
                        )

                        # <---- 修改: 分开记录表格
                        # 记录干净表格
                        logger.log({
                            "eval/clean_confusion_matrix_table": clean_table,
                        }, step=epoch)
                        # 记录毒化表格
                        logger.log({
                            "eval/poison_confusion_matrix_table": poison_table
                        }, step=epoch)
                    
                    # 对于tensorboard，额外记录混淆矩阵数据为文本
                    elif logger.log_type == 'tensorboard':
                        # 由于TensorBoard不支持表格，可以将混淆矩阵作为文本记录
                        logger.writer.add_text('eval/clean_confusion_matrix_data', str(clean_conf_matrix), epoch)
                        logger.writer.add_text('eval/poison_confusion_matrix_data', str(poison_conf_matrix), epoch)
                    
                except Exception as e:
                    print(f"无法创建或记录混淆矩阵可视化: {e}")
                    import traceback
                    traceback.print_exc()
    
    return clean_acc, poison_acc, asr

