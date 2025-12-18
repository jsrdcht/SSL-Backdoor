"""
PatchSearch防御方法的毒药分类器实现。

该模块实现了一个基于ResNet的毒药分类器，用于区分正常样本和包含后门触发器的样本。
"""

import os
import math
import random
import logging
import numpy as np
from functools import partial
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, ResNet
from PIL import Image
from tqdm import tqdm

from .utils.dataset import get_transforms
from .utils.evaluation import AverageMeter, ProgressMeter, accuracy
from .utils.visualization import denormalize, show_images_grid
from ssl_backdoor.utils.utils import set_seed


class PoisonDataset(Dataset):
    """
    用于训练毒药分类器的数据集，可以生成干净和掺毒的样本。
    """
    def __init__(self, args, path_to_txt_file, pre_transform, post_transform, poison_dir, topk_poisons, output_type='clean'):
        self.output_type = output_type
        self.args = args

        # 加载顶部毒药补丁
        self.poisons = []
        for i in range(topk_poisons):
            poison_file = os.path.join(poison_dir, f'{i:05d}.png')
            if os.path.exists(poison_file):
                self.poisons.append(Image.open(poison_file).convert('RGB'))
        
        self.poisons = self.poisons[:topk_poisons]
        if len(self.poisons) == 0:
            raise ValueError(f"在{poison_dir}中未找到毒药补丁图像")

        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def paste_poison(self, img):
        """将毒药补丁粘贴到图像上"""
        if 'imagenet' in self.args.dataset_name:
            margin = 10
            image_size = 224
            poison_size_low, poison_size_high = 20, 80
        elif 'cifar' in self.args.dataset_name:
            margin = 2
            image_size = 32
            poison_size_low, poison_size_high = 4, 16
        elif 'stl' in self.args.dataset_name:
            margin = 5
            image_size = 96
            poison_size_low, poison_size_high = 12, 40
        else:
            raise ValueError(f'意外的数据集: {self.args.dataset_name}')
        
        # 随机选择一个毒药补丁
        poison = self.poisons[np.random.randint(low=0, high=len(self.poisons))]
        
        # 随机调整毒药大小
        new_s = np.random.randint(low=poison_size_low, high=poison_size_high)
        poison = poison.resize((new_s, new_s))
        
        # 随机放置位置
        loc_box = (margin, image_size - (new_s + margin))
        loc_h, loc_w = np.random.randint(*loc_box), np.random.randint(*loc_box)
        img.paste(poison, (loc_h, loc_w))
        return img

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        is_poisoned = 'poison' in image_path
        img = Image.open(image_path).convert('RGB')
        is_poison = np.random.rand() > 0.5

        if self.output_type == 'clean' or (self.output_type == 'rand' and not is_poison):
            target = 0
            img = self.pre_transform(img)
            img = self.post_transform(img)
        elif self.output_type == 'poisoned' or (self.output_type == 'rand' and is_poison):
            target = 1
            img = self.pre_transform(img)
            img = self.paste_poison(img)
            img = self.post_transform(img)
        else:
            raise ValueError(f'意外的output_type: {self.output_type}')

        return image_path, img, target, is_poisoned, idx

    def __len__(self):
        return len(self.file_list)


class ValPoisonDataset(Dataset):
    """
    用于验证毒药分类器的数据集，使用已标记的正负样本。
    """
    def __init__(self, path_to_txt_file, pos_inds, neg_inds, transform):
        with open(path_to_txt_file, 'r') as f:
            file_list = f.readlines()
            file_list = [row.strip().split() for row in file_list]

        pos_samples = [(file_list[i][0], 1) for i in pos_inds]
        neg_samples = [(file_list[i][0], 0) for i in neg_inds]
        self.samples = pos_samples + neg_samples
        self.transform = transform

    def __getitem__(self, idx):
        image_path, target = self.samples[idx]
        is_poisoned = 'poison' in image_path
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)

        return image_path, img, target, is_poisoned, idx

    def __len__(self):
        return len(self.samples)


class EnsembleNet(nn.Module):
    """多模型集成网络，用于提高分类准确率和鲁棒性"""
    def __init__(self, models):
        super(EnsembleNet, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        y = torch.stack([model(x) for model in self.models], dim=0)
        y = torch.einsum('kbd->bkd', y)
        y = y.mean(dim=1)
        return y


def worker_init_fn(baseline_seed, it, worker_id):
    """为数据加载器的每个工作进程设置随机种子"""
    np.random.seed(baseline_seed + it + worker_id)


def prepare_datasets(args, poison_scores):
    """
    准备训练和验证数据集
    
    参数:
        args: 配置参数
        poison_scores: 毒性分数数组
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    logger = logging.getLogger('patchsearch')
    
    # 获取数据集的归一化参数和变换
    if args.dataset_name == 'imagenet100':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        args.image_size = 224
    elif args.dataset_name == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])
        args.image_size = 32
    elif args.dataset_name == 'stl10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        args.image_size = 96
    else:
        raise ValueError(f"未知数据集 '{args.dataset_name}'")
    
    # 定义训练数据增强
    train_t1 = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
    ])
    train_t2 = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # 创建训练数据集
    train_dataset = PoisonDataset(
        args,
        args.train_file,
        pre_transform=train_t1, 
        post_transform=train_t2,
        poison_dir=args.poison_dir,
        topk_poisons=args.topk_poisons,
        output_type='rand',
    )

    # 保存样本图像以便检查
    inds = np.random.randint(low=0, high=len(train_dataset), size=40)
    
    # 保存干净图像样本
    train_dataset.output_type = 'clean'
    clean_images = torch.stack([train_dataset[i][1] for i in inds])
    show_images_grid(clean_images, args.output_dir, 'Train_Clean_Images', args)
    
    # 保存掺毒图像样本
    train_dataset.output_type = 'poisoned'
    poisoned_images = torch.stack([train_dataset[i][1] for i in inds])
    show_images_grid(poisoned_images, args.output_dir, 'Train_Poisoned_Images', args)
    
    # 恢复为随机模式
    train_dataset.output_type = 'rand'
    rand_images = torch.stack([train_dataset[i][1] for i in inds])
    show_images_grid(rand_images, args.output_dir, 'Train_Rand_Images', args)

    # 根据毒性分数选择正负样本索引
    sorted_inds = (-poison_scores).argsort()
    pos_inds = sorted_inds[:args.topk_poisons]  # 最可能的毒药样本
    neg_inds = sorted_inds[-args.topk_poisons:]  # 最不可能的毒药样本
    
    # 选择训练集，删除可能的极端样本
    train_inds = sorted_inds[int(args.top_p*len(train_dataset)):-args.topk_poisons]
    logger.info(f'训练数据集大小: {len(train_inds)/1000:.1f}k')

    # 取训练数据集的子集
    train_dataset.output_type = 'rand'
    train_dataset = Subset(train_dataset, train_inds)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True,
        worker_init_fn=partial(worker_init_fn, args.seed, 0)
    )

    # 创建验证数据集
    if 'imagenet' in args.dataset_name:
        val_t1 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
    elif 'cifar' in args.dataset_name:
        val_t1 = transforms.Compose([
            transforms.Resize(32),
        ])
    elif 'stl' in args.dataset_name:
        val_t1 = transforms.Compose([
            transforms.Resize(96),
        ])
    else:
        raise ValueError(f'意外的数据集: {args.dataset_name}')
        
    val_t2 = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    val_dataset = ValPoisonDataset(
        args.train_file,
        pos_inds=pos_inds,
        neg_inds=neg_inds,
        transform=transforms.Compose([val_t1, val_t2]),
    )
    
    # 保存验证图像样本
    val_images = torch.stack([val_dataset[i][1] for i in range(min(40, len(val_dataset)))])
    show_images_grid(val_images, args.output_dir, 'Val_Images', args)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True,
        worker_init_fn=partial(worker_init_fn, args.seed, 0)
    )

    # 创建测试数据集（用于最终评估）
    test_dataset = PoisonDataset(
        args,
        args.train_file,
        pre_transform=val_t1, 
        post_transform=val_t2,
        poison_dir=args.poison_dir,
        topk_poisons=args.topk_poisons,
        output_type='clean'
    )
    
    # 保存测试图像样本
    inds = np.random.randint(low=0, high=len(test_dataset), size=min(40, len(test_dataset)))
    test_images = torch.stack([test_dataset[i][1] for i in inds])
    show_images_grid(test_images, args.output_dir, 'Test_Images', args)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def validate(val_loader, model, args):
    """
    在验证集上评估模型性能
    
    参数:
        val_loader: 验证数据加载器
        model: 模型
        args: 配置参数
    
    返回:
        recall: 召回率
        precision: 精度
        f1_beta: F1分数
    """
    logger = logging.getLogger('patchsearch')
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time],
        prefix='验证'
    )

    # 切换到评估模式
    model.eval()

    pred_is_poison = np.zeros(len(val_loader.dataset))
    gt_is_poison = np.zeros(len(val_loader.dataset))

    end = time.time()
    for i, (_, images, target, _, inds) in enumerate(val_loader):
        if i == 0:
            show_images_grid(images, args.output_dir, f'eval-images-iteration-0', args)

        # 测量数据加载时间
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)

        # 计算输出
        with torch.no_grad():
            output = model(images)

        pred = output.argmax(dim=1).detach().cpu()
        pred_is_poison[inds.numpy()] = pred.numpy()
        gt_is_poison[inds.numpy()] = target.numpy().astype(int)

        # 测量经过的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))

    # 计算召回率（检测到的毒药/所有真实毒药）
    if np.sum(gt_is_poison) > 0:
        recall = pred_is_poison[np.where(gt_is_poison)[0]].astype(float).mean()
    else:
        recall = 0.0
    logger.info(f'毒药召回率: {recall*100:.1f}%')

    # 计算精度（真实毒药/所有检测到的毒药）
    if np.sum(pred_is_poison) > 0:
        precision = gt_is_poison[np.where(pred_is_poison)[0]].astype(float).mean()
    else:
        precision = 0.0
    logger.info(f'毒药精度: {precision*100:.1f}%')

    # 计算F1分数
    beta = 1
    if precision > 0 or recall > 0:
        f1_beta = (1 + beta**2) * (precision * recall) / ((beta**2) * precision + recall + 1e-10)
    else:
        f1_beta = 0.0
    logger.info(f'毒药F1_beta分数 (beta = {beta}): {f1_beta*100:.1f}%')

    if math.isnan(recall) or math.isnan(precision) or math.isnan(f1_beta):
        return 0., 0., 0.

    return recall, precision, f1_beta


def test(test_loader, model, args):
    """
    在测试集上评估模型并生成过滤后的干净数据集
    
    参数:
        test_loader: 测试数据加载器
        model: 模型
        args: 配置参数
    
    返回:
        poison_recall: 毒药召回率
        poison_precision: 毒药精度
        pred_is_poison: 模型预测的毒药样本标记
    """
    logger = logging.getLogger('patchsearch')
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time],
        prefix='测试'
    )

    # 切换到评估模式
    model.eval()

    pred_is_poison = np.zeros(len(test_loader.dataset))
    prob_is_poison = np.zeros(len(test_loader.dataset))
    gt_is_poison = np.zeros(len(test_loader.dataset))

    end = time.time()
    for i, (_, images, _, is_poisoned, inds) in enumerate(test_loader):
        # 测量数据加载时间
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)

        # 计算输出
        with torch.no_grad():
            output = model(images)
            probs = F.softmax(output, dim=1)

        pred = output.argmax(dim=1).detach().cpu()
        pred_is_poison[inds.numpy()] = pred.numpy()
        prob_is_poison[inds.numpy()] = probs[:, 1].detach().cpu().numpy()
        gt_is_poison[inds.numpy()] = is_poisoned.numpy().astype(int)

        # 测量经过的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (len(test_loader) // 20) == 0:
            logger.info(progress.display(i))

    logger.info(f'需要移除的总毒药数: {np.count_nonzero(pred_is_poison)}')
    
    # Calculate metrics
    if len(np.unique(gt_is_poison)) > 1:
        tn, fp, fn, tp = confusion_matrix(gt_is_poison, pred_is_poison).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        try:
            auroc = roc_auc_score(gt_is_poison, prob_is_poison)
        except ValueError:
            auroc = 0.0
    else:
        # If only one class is present in ground truth
        tpr = 0.0
        fpr = 0.0
        precision = 0.0
        auroc = 0.0
        logger.warning("Ground truth only contains one class. Metrics might be invalid.")
        
    logger.info(f'检测指标:')
    logger.info(f'TPR (Recall): {tpr*100:.2f}%')
    logger.info(f'FPR: {fpr*100:.2f}%')
    logger.info(f'Precision: {precision*100:.2f}%')
    logger.info(f'AUROC: {auroc*100:.2f}%')

    return tpr, precision, pred_is_poison


def train(args, poison_scores, external_test_loader=None):
    """
    训练一个集成分类器来检测和过滤毒药样本
    
    参数:
        args: 配置参数
        poison_scores: 毒性分数数组
        external_test_loader: 外部测试数据加载器
        
    返回:
        filtered_file_path: 过滤后的干净数据集文件路径
    """
    # 设置随机种子以确保可重现性
    if args.seed is not None:
        set_seed(args.seed)

    
    # 设置日志器
    logger = logging.getLogger('patchsearch')
    logger.info(f"开始训练毒药分类器")
    
    # 记录参数
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    # 准备数据集
    train_loader, val_loader, test_loader = prepare_datasets(args, poison_scores)
    
    # 创建集成模型
    models = []
    for model_i in range(args.model_count):
        logger.info('='*40 + f' 模型 {model_i} ' + '='*40)
        
        # 更新数据加载器的随机种子
        train_loader.worker_init_fn = partial(worker_init_fn, args.seed, model_i)
        val_loader.worker_init_fn = partial(worker_init_fn, args.seed, model_i)
        
        # 创建小型ResNet模型
        model = ResNet(block=BasicBlock, layers=[1, 1, 1, 1])
        model.fc = nn.Linear(512, 2)  # 二分类：干净/毒药
        model = model.cuda()
        
        # 设置优化器
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            args.max_iterations
        )
        
        # 训练指标
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        lrs = AverageMeter('LR', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            args.max_iterations,
            [batch_time, data_time, lrs, losses, top1],
            prefix="训练: "
        )
        
        # 切换到训练模式
        model.train()
        
        # 训练循环
        it = 0
        val_metrics = []
        while it < args.max_iterations:
            end = time.time()
            for _, images, target, is_poisoned, inds in train_loader:
                if it >= args.max_iterations:
                    break
                    
                # 保存前几个批次的图像以便检查
                if it < 5:
                    show_images_grid(images, args.output_dir, f'train-images-iteration-{it:05d}', args)
                
                # 测量数据加载时间
                data_time.update(time.time() - end)
                
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                
                # 前向传播
                output = model(images)
                
                # 计算损失
                loss = F.cross_entropy(output, target)
                losses.update(loss.item(), images.size(0))
                
                # 计算准确率
                acc1 = accuracy(output, target, topk=(1,))[0]
                top1.update(acc1[0], images.size(0))
                
                # 更新学习率
                lrs.update(lr_scheduler.get_last_lr()[-1])
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 测量经过的时间
                batch_time.update(time.time() - end)
                end = time.time()
                
                # 打印训练过程
                if it % args.print_freq == 0:
                    logger.info(progress.display(it))
                
                # 定期在验证集上评估
                if it % args.eval_freq == 0:
                    recall, precision, f1_beta = validate(val_loader, model, args)
                    val_metrics.append((recall, precision, f1_beta))
                    
                    # 早停：如果最近10次验证的F1分数相同，则停止训练
                    vm = set([int(x[2]*100) for x in val_metrics[-10:]])
                    logger.info(f"最近10次验证的F1分数: {vm}")
                    if len(vm) == 1 and len(val_metrics) > 10:
                        logger.info("F1分数稳定，提前停止训练")
                        it = args.max_iterations
                        break
                        
                    # 切换回训练模式
                    model.train()
                
                # 更新学习率
                lr_scheduler.step()
                
                it += 1
                
            # 将训练好的模型添加到集合中
        models.append(model)
    
    # 创建集成模型
    ensemble_model = EnsembleNet(models)
    
    # 在测试集上进行最终评估
    logger.info(f'在测试数据上运行推理')
    recall, precision, preds = test(test_loader, ensemble_model, args)

    if external_test_loader is not None:
        logger.info(f'在外部测试数据上运行推理 (Balanced Clean + Poisoned)')
        ext_recall, ext_precision, ext_preds = test(external_test_loader, ensemble_model, args)
        
        # Calculate extra metrics if needed, test() already prints TPR/FPR/AUROC
        # But we might want to ensure they are logged clearly as "External Test Metrics"
        logger.info(f'外部测试集评估完成')
    
    # 保存过滤后的干净数据集
    filtered_file_path = os.path.join(args.output_dir, 'filtered.txt')
    with open(filtered_file_path, 'w') as f:
        for line, is_poisoned in zip(test_loader.dataset.file_list, preds):
            if not is_poisoned:  # 只保留干净样本
                f.write(f'{line}\n')
    
    logger.info(f"过滤后的数据集已保存到: {filtered_file_path}")
    
    return filtered_file_path


def run_poison_classifier(
    poison_scores,
    output_dir,
    train_file,
    poison_dir,
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
    seed=42,
    external_test_loader=None
):
    """
    运行毒药分类器以过滤可能的后门样本
    
    参数:
        poison_scores: 毒性分数数组
        output_dir: 输出目录
        train_file: 训练文件路径
        poison_dir: 包含顶部毒药补丁的目录
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
    # 设置参数
    class Args:
        pass
    
    args = Args()
    args.output_dir = output_dir
    args.train_file = train_file
    args.poison_dir = poison_dir
    args.dataset_name = dataset_name
    args.topk_poisons = topk_poisons
    args.top_p = top_p
    args.model_count = model_count
    args.max_iterations = max_iterations
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.lr = lr
    args.momentum = momentum
    args.weight_decay = weight_decay
    args.print_freq = print_freq
    args.eval_freq = eval_freq
    args.seed = seed
    
    # 创建输出目录
    # 使用descriptive的目录名
    dir_name = f'poison_classifier_topk_{topk_poisons}_ensemble_{model_count}_max_iter_{max_iterations}'
    output_dir = os.path.join(output_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
    # 设置日志器
    logger = logging.getLogger('patchsearch')
    
    # 训练模型并过滤数据集
    filtered_file_path = train(args, poison_scores, external_test_loader)
    
    return filtered_file_path

import time  # 添加缺失的导入 