import os
import sys
import argparse
import random
import time
import socket
import warnings
import math
import builtins
from pathlib import Path
import importlib.util

# 第三方库
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# 项目内部模块
import ssl_backdoor.ssl_trainers.moco.loader
import ssl_backdoor.ssl_trainers.moco.builder
import ssl_backdoor.ssl_trainers.simsiam.builder
import ssl_backdoor.ssl_trainers.byol.builder
import ssl_backdoor.ssl_trainers.simclr.builder
import ssl_backdoor.ssl_trainers.utils as utils
import ssl_backdoor.datasets.dataset

from ssl_backdoor.ssl_trainers.utils import (
    initialize_distributed_training, 
    load_config_from_yaml, 
    merge_configs,
    Logger, 
    load_config, 
    adjust_learning_rate
)

# 测试功能
# from tools.test import test_model  # <-- 删除这一行


def get_trainer(config_or_path):
    """
    获取MoCo训练器函数
    
    Args:
        config_or_path: 配置字典或配置文件路径
        
    Returns:
        训练函数，可直接调用启动训练
    """
    # 处理配置
    if isinstance(config_or_path, str):
        config = load_config(config_or_path)
    elif isinstance(config_or_path, argparse.Namespace):
        config = vars(config_or_path)
    elif isinstance(config_or_path, dict):
        config = config_or_path
    else:
        raise TypeError("config_or_path 必须是 str, dict 或 argparse.Namespace")

    # 创建一个 argparse.Namespace 对象来存储配置
    args = argparse.Namespace(**config)
    
    # 设置默认值
    args.save_folder_root = getattr(args, 'save_folder_root', 'checkpoints')
    args.experiment_id = getattr(args, 'experiment_id', f"{args.method}_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}")
    args.logger_type = getattr(args, 'logger_type', 'tensorboard')
    args.save_folder = os.path.join(args.save_folder_root, args.experiment_id)
    
    # 更新配置字典以保持一致性
    config['experiment_id'] = args.experiment_id
    config['logger_type'] = args.logger_type
    
    # 创建保存目录
    os.makedirs(args.save_folder, exist_ok=True)
    
    # 设置分布式训练参数
    args.gpus = list(range(torch.cuda.device_count()))
    args.world_size = len(args.gpus)
    args.rank = 0
    args.distributed = getattr(args, 'multiprocessing_distributed', False)
    config['distributed'] = args.distributed

    # 保存最终配置
    try:
        with open(os.path.join(args.save_folder, 'final_config.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        print(f"警告：无法保存最终配置文件。错误: {e}")
    
    # 返回训练启动函数
    def train_func():
        # 打印最终使用的Namespace对象
        print("\n传递给 main_worker 的 args 对象:")
        try:
            import pprint
            pprint.pprint(vars(args))
        except ImportError:
            print(args)
        print("-"*30)

        
        if args.distributed:
            mp.spawn(main_worker, nprocs=len(args.gpus), args=(args,))
        else:
            main_worker(0, args)
            
    return train_func



def main_worker(index, args):
    # args 现在是一个 Namespace 对象，代码应该可以正常工作
    initialize_distributed_training(args, index)

    # 初始化日志记录器（主进程）
    global logger
    if index == 0:
        # 初始化日志记录器
        args.enable_logging = args.logger_type.lower() != 'none'
        logger = Logger(
            log_type=args.logger_type,
            save_dir=args.save_folder,
            experiment_id=args.experiment_id,
            config=vars(args),
            project=f"ssl-backdoor"
        )
    else:
        args.enable_logging = False
        logger = None

    # 初始化混合精度训练
    scaler = GradScaler(enabled=args.amp) if hasattr(args, 'amp') and args.amp else None

    # 在分布式环境中抑制非主进程的打印
    if args.multiprocessing_distributed and args.index != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    print(f"使用GPU: {args.gpus} 在 '{socket.gethostname()}' 上训练")

    # 设置随机种子
    if args.seed is not None:
        args.seed = args.seed + args.rank
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    cudnn.deterministic = True
    cudnn.benchmark = True

    # 创建模型
    print(f"=> 创建模型 '{args.arch}'")
    if args.method == 'moco':
        model = ssl_backdoor.ssl_trainers.moco.builder.MoCo(
            models.__dict__[args.arch], args.feature_dim, args.moco_k, args.moco_m, 
            contr_tau=args.moco_contr_tau,
            align_alpha=args.moco_align_alpha, unif_t=args.moco_unif_t,
            dataset=args.dataset)
    elif args.method == 'simsiam':
        model = ssl_backdoor.ssl_trainers.simsiam.builder.SimSiam(
            models.__dict__[args.arch], dim=args.feature_dim, pred_dim=args.pred_dim, dataset=args.dataset)
    elif args.method == 'byol':
        model = ssl_backdoor.ssl_trainers.byol.builder.BYOL(
            models.__dict__[args.arch], 
            dim=args.feature_dim, 
            proj_dim=getattr(args, 'proj_dim', None),
            pred_dim=getattr(args, 'pred_dim', None),
            tau=getattr(args, 'byol_tau', 0.99),
            dataset=args.dataset)
    elif args.method == 'simclr':
        model = ssl_backdoor.ssl_trainers.simclr.builder.SimCLR(
            models.__dict__[args.arch], 
            dim=args.feature_dim, 
            proj_dim=getattr(args, 'proj_dim', 128),
            dataset=args.dataset)
    else:
        raise ValueError(f"未知方法 '{args.method}'")

    model.cuda(args.gpu)

    # 分布式训练设置
    if args.distributed:
        if args.method == 'simsiam' or args.method == 'byol':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # WandB模型监控（仅主进程）
    if index == 0 and args.logger_type == 'wandb' and logger is not None and logger.writer is not None:
        logger.writer.watch(model, log="all", log_freq=args.print_freq)

    # 设置优化器参数
    if hasattr(args, 'fix_pred_lr') and args.fix_pred_lr: # only simsiam needs this
        if args.method == 'simsiam':
            optim_params = [
                {'params': model.module.encoder.parameters(), 'fix_lr': False},
                {'params': model.module.projector.parameters(), 'fix_lr': False},
                {'params': model.module.predictor.parameters(), 'fix_lr': True}
            ]
        else:
            optim_params = model.parameters()
    else:
        optim_params = model.parameters()

    # 根据模型架构选择优化器
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            optim_params, args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"未知优化器: '{args.optimizer}'")

    # 创建学习率调度器
    if hasattr(args, 'lr_schedule'):
        # 检查是否有参数组需要固定学习率
        has_fixed_lr_groups = any('fix_lr' in pg and pg['fix_lr'] for pg in optimizer.param_groups)
        print(f"has_fixed_lr_groups: {has_fixed_lr_groups}")
        
        if args.lr_schedule.lower() == 'cos':
            if has_fixed_lr_groups:
                # 为每个参数组创建单独的调度函数
                lr_lambdas = []
                for param_group in optimizer.param_groups:
                    if 'fix_lr' in param_group and param_group['fix_lr']:
                        # 对于固定学习率的参数组，lambda函数总是返回1.0
                        lr_lambdas.append(lambda _: 1.0)
                    else:
                        # 对于非固定学习率的参数组，使用余弦调度
                        lr_lambdas.append(lambda epoch: 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)))
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
                print(f"=> 使用带有固定学习率支持的余弦退火调度 (T_max={args.epochs})")
            else:
                # 没有固定学习率的参数组，使用标准CosineAnnealingLR
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=0
                )
                print(f"=> 使用余弦退火学习率调度 (T_max={args.epochs})")
                
        elif args.lr_schedule.lower() == 'step':
            milestones = args.lr_drops if hasattr(args, 'lr_drops') else [args.epochs // 2]
            gamma = args.lr_drop_gamma if hasattr(args, 'lr_drop_gamma') else 0.1
            
            if has_fixed_lr_groups:
                raise ValueError("阶梯式学习率调度(step)不支持固定学习率参数组(fix_lr)")
            else:
                # 没有固定学习率的参数组，使用标准MultiStepLR
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=milestones, gamma=gamma
                )
                print(f"=> 使用阶梯式学习率调度 (milestones={milestones}, gamma={gamma})")
        else:
            print(f"警告: 未知的学习率调度类型 '{args.lr_schedule}'，将使用常数学习率")
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        print("警告: 未设置学习率调度类型，将使用常数学习率")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    # 从检查点恢复（如果有）
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> 加载检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device('cuda', args.gpu))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # 如果检查点中有scheduler状态，则恢复
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("=> 已恢复学习率调度器状态")
            # 如果检查点中有scaler状态并且启用了AMP，则恢复
            if 'scaler' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler'])
                print("=> 已恢复 GradScaler 状态")
            print(f"=> 已加载检查点 '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> 未找到检查点 '{args.resume}'")

    # 创建数据加载器
    train_loader = create_data_loader(args)

    # 检查是否启用评估
    do_eval = hasattr(args, 'test_config') and isinstance(args.test_config, dict)

    
    if do_eval and args.rank == 0:
        print(f"模型评估已启用，评估频率：每 {args.eval_frequency} 个 epoch")
    
    # 追踪最佳结果
    best_results = {
        'clean_acc': 0.0,
        'poison_acc': 0.0,
        'asr': 0.0,
        'epoch': 0
    }

    # 主训练循环
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # 训练一个epoch
        train(train_loader, model, optimizer, epoch, args, scaler)
        
        # 更新学习率调度器
        scheduler.step()
        
        # 记录当前学习率（主进程）
        if args.rank == 0 and logger is not None:
            current_lr = optimizer.param_groups[0]['lr']
            logger.add_scalar('train/learning_rate', current_lr, epoch)
            print(f"Epoch {epoch} 的学习率: {current_lr:.8f}")
        
        # 确定是否需要保存检查点和执行评估
        should_save = (epoch + 1) % args.save_freq == 0
        should_eval = do_eval and (epoch + 1) % args.eval_frequency == 0
        
        # 保存检查点和执行评估（主进程）
        save_filename = None
        if (args.distributed and args.rank == 0) or (args.index == 0):
            # 保存检查点（如果需要）
            if should_save or should_eval:
                if should_save:
                    # 正常保存训练检查点
                    save_dir = args.save_folder
                    save_filename = os.path.join(save_dir, f'checkpoint_{epoch:04d}.pth.tar')
                    save_dict = {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }
                    # 如果启用了AMP，也保存scaler的状态
                    if scaler is not None:
                        save_dict['scaler'] = scaler.state_dict()
                    torch.save(save_dict, save_filename)
                    print(f"已保存检查点至 '{save_filename}'")
                elif should_eval:
                    # 仅为评估创建临时检查点，保存在/tmp目录下
                    tmp_dir = os.path.join('/tmp', f'ssl_backdoor_eval_{args.experiment_id}')
                    os.makedirs(tmp_dir, exist_ok=True)
                    save_filename = os.path.join(tmp_dir, f'tmp_checkpoint_{epoch:04d}.pth.tar')
                    save_dict = {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }
                    # 如果启用了AMP，也保存scaler的状态
                    if scaler is not None:
                        save_dict['scaler'] = scaler.state_dict()
                    torch.save(save_dict, save_filename)
                    print(f"已保存临时评估检查点至 '{save_filename}'")
        
        # 确保所有进程同步
        if args.distributed:
            dist.barrier()
        
        # 执行评估
        if should_eval:
            # 在分布式环境中同步文件路径
            if args.distributed:
                if args.rank == 0:
                    object_list = [save_filename]
                else:
                    object_list = [None]
                
                # 广播保存的文件路径
                dist.broadcast_object_list(object_list, src=0)
                save_filename = object_list[0]
            
            # --- 将导入移动到这里 ---
            from tools.test import test_model
            # ---------------------
            
            # 执行评估
            print(f"rank {args.rank} 正在评估 epoch {epoch+1} 的模型...")
            clean_acc, poison_acc, asr = test_model(save_filename, epoch + 1, args, logger, config=(args.test_config if hasattr(args, 'test_config') else None))
            
            # 如果是临时评估检查点，测试完成后删除
            if (args.distributed and args.rank == 0) or (args.index == 0):
                if should_eval and not should_save and os.path.exists(save_filename):
                    try:
                        os.remove(save_filename)
                        print(f"已删除临时评估检查点: {save_filename}")
                        # 尝试删除临时目录（如果为空）
                        tmp_dir = os.path.dirname(save_filename)
                        if os.path.exists(tmp_dir) and not os.listdir(tmp_dir):
                            os.rmdir(tmp_dir)
                            print(f"已删除空的临时目录: {tmp_dir}")
                    except Exception as e:
                        print(f"删除临时评估检查点时出错: {e}")
            
            # 记录评估结果（主进程）
            if args.rank == 0:
                # 添加到日志
                if logger is not None:
                    logger.add_scalar('eval/clean_acc', clean_acc, epoch + 1)
                    logger.add_scalar('eval/poison_acc', poison_acc, epoch + 1)
                    logger.add_scalar('eval/attack_success_rate', asr, epoch + 1)
                    
                    # 构造日志消息
                    log_message = f"评估结果 - Epoch {epoch+1} | Clean Acc: {clean_acc:.2f}% | Poison Acc: {poison_acc:.2f}% | Attack Success Rate: {asr:.2f}%"
                    print(log_message)

                    logger.log({
                        "eval/epoch": epoch + 1,
                        "eval/clean_acc": clean_acc,
                        "eval/poison_acc": poison_acc,
                        "eval/attack_success_rate": asr,
                        "eval/summary": log_message
                    })
                
                # 更新最佳结果
                if clean_acc > best_results['clean_acc']:
                    best_results['clean_acc'] = clean_acc
                    best_results['poison_acc'] = poison_acc
                    best_results['asr'] = asr
                    best_results['epoch'] = epoch + 1
                
                # 打印当前和最佳结果
                print(f"当前评估: Clean Acc: {clean_acc:.2f}%, Poison Acc: {poison_acc:.2f}%, ASR: {asr:.2f}%")
                print(f"最佳评估: Clean Acc: {best_results['clean_acc']:.2f}%, "
                      f"Poison Acc: {best_results['poison_acc']:.2f}%, "
                      f"ASR: {best_results['asr']:.2f}% (Epoch {best_results['epoch']})")
                
                # 创建评估摘要文件
                with open(os.path.join(args.save_folder, 'eval_summary.txt'), 'w') as f:
                    f.write(f"实验ID: {args.experiment_id}\n")
                    f.write(f"最佳 Clean Accuracy: {best_results['clean_acc']:.2f}% (Epoch {best_results['epoch']})\n")
                    f.write(f"对应 Poison Accuracy: {best_results['poison_acc']:.2f}%\n")
                    f.write(f"对应 Attack Success Rate: {best_results['asr']:.2f}%\n")
                    f.write(f"最后评估 (Epoch {epoch+1}):\n")
                    f.write(f"  Clean Accuracy: {clean_acc:.2f}%\n")
                    f.write(f"  Poison Accuracy: {poison_acc:.2f}%\n")
                    f.write(f"  Attack Success Rate: {asr:.2f}%\n")
        
        # 支持提前退出
        if args.ablation and epoch == 99:
            exit(0)

    # 关闭日志
    if args.index == 0 and logger is not None:
        logger.close()


def create_data_loader(args):
    """
    根据配置创建数据加载器
    
    Args:
        args: 配置参数
        
    Returns:
        torch.utils.data.DataLoader: 训练数据加载器
    """
    # 数据集参数配置
    dataset_params = {
        'imagenet100': {
            'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            'image_size': 224
        },
        'cifar10': {
            'normalize': transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]), 
            'image_size': 32
        },
        'stl10': {
            'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            'image_size': 96
        },
    }

    if args.dataset not in dataset_params:
        raise ValueError(f"不支持的数据集: '{args.dataset}'")

    # 设置图像大小参数
    params = dataset_params[args.dataset]
    args.image_size = params['image_size']

    # 获取最小crop比例，默认为0.2
    min_scale = getattr(args, 'min_crop_scale', 0.2)
    print(f"使用的RandomResizedCrop最小缩放比例: {min_scale}")

    # 定义数据增强
    augmentation = [
        transforms.RandomResizedCrop(args.image_size, scale=(min_scale, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([ssl_backdoor.ssl_trainers.moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        params['normalize'],
    ]

    # 创建双视图转换
    composed_transforms = ssl_backdoor.ssl_trainers.moco.loader.TwoCropsTransform(
        transforms.Compose(augmentation)
    )

    # 支持的数据集类
    dataset_classes = {
        'bp': ssl_backdoor.datasets.dataset.BPTrainDataset,
        'corruptencoder': ssl_backdoor.datasets.dataset.CorruptEncoderTrainDataset,
        'sslbkd': ssl_backdoor.datasets.dataset.SSLBackdoorTrainDataset,
        'ctrl': ssl_backdoor.datasets.dataset.CTRLTrainDataset,
        'clean': ssl_backdoor.datasets.dataset.FileListDataset,
        'blto': ssl_backdoor.datasets.dataset.BltoPoisoningPoisonedTrainDataset,
        'optimized': ssl_backdoor.datasets.dataset.OptimizedTrainDataset,
    }

    if args.attack_algorithm not in dataset_classes:
        raise ValueError(f"不支持的攻击算法: '{args.attack_algorithm}'")

    # 创建数据集
    train_dataset = dataset_classes[args.attack_algorithm](args, args.data, composed_transforms)

    # 设置分布式采样器
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

    # 创建并返回数据加载器
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )


def train(train_loader, model, optimizer, epoch, args, scaler):
    batch_time = utils.AverageMeter('Time', '6.3f')
    data_time = utils.AverageMeter('Data', '6.3f')

    # 根据数据集设置反归一化变换
    dataset_norm_params = {
        'imagenet100': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'cifar10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
        'stl10': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    }
    
    if args.dataset not in dataset_norm_params:
        raise ValueError(f"不支持的数据集: '{args.dataset}'")
    
    # 设置反归一化变换
    mean = dataset_norm_params[args.dataset]['mean']
    std = dataset_norm_params[args.dataset]['std']
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)], 
        std=[1/s for s in std]
    )
    inv_transform = transforms.Compose([inv_normalize, transforms.ToPILImage()])
    
    # 创建训练图像保存目录
    img_save_dir = os.path.join(args.save_folder, "train_images")
    os.makedirs(img_save_dir, exist_ok=True)
    img_ctr = 0

    contr_meter = utils.AverageMeter('Contr-Loss', '.4e')
    
    # 根据方法设置监控指标
    if args.method == 'moco':
        acc1 = utils.AverageMeter('Contr-Acc1', '6.2f')
        acc5 = utils.AverageMeter('Contr-Acc5', '6.2f')
        loss_meters = [contr_meter, acc1, acc5, utils.ProgressMeter.BR]
    elif args.method == 'simsiam' or args.method == 'byol':
        loss_meters = [contr_meter, utils.ProgressMeter.BR]
    else:
        loss_meters = [contr_meter]

    # 移除行尾的BR
    if loss_meters and loss_meters[-1] == utils.ProgressMeter.BR:
        loss_meters = loss_meters[:-1]

    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time] + loss_meters,
        prefix=f"Epoch: [{epoch}]"
    )

    # 切换到训练模式
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # 测量数据加载时间
        data_time.update(time.time() - end)

        # 将图像移动到GPU
        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # 在第一个epoch的前20个batch保存目标类的图像样本
        if epoch == 0 and i < 20:
            # 获取debug目标或第一个攻击目标
            debug_target = getattr(args, 'debug_target', None)
            if debug_target is not None:
                debug_target = int(debug_target)
            elif hasattr(args, 'attack_target_list'):
                debug_target = args.attack_target_list[0]
            
            # 保存目标类图像
            for batch_index in range(images[0].size(0)):
                if debug_target is not None and int(target[batch_index].item()) == debug_target:
                    img_ctr += 1
                    # 保存两个视图
                    for view_idx, view in enumerate(images[:2]):
                        inv_image = inv_transform(view[batch_index].cpu())
                        save_path = os.path.join(img_save_dir, f"{img_ctr:05d}_view_{view_idx}.png")
                        inv_image.save(save_path)

        # 根据是否使用混合精度训练选择计算方式
        if args.amp:
            with autocast():
                loss = model(images[0], images[1])
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss = model(images[0], images[1])
            loss.backward()
            optimizer.step()

        # 如果是BYOL，每个batch后更新目标网络
        if args.method == 'byol' and hasattr(model, 'module'):
            # 为分布式模型设置进度
            model.module.update_target(float(epoch) / args.epochs)
        elif args.method == 'byol':
            # 为非分布式模型设置进度
            model.update_target(float(epoch) / args.epochs)

        # 记录损失
        if args.index == 0:
            bs = images[0].shape[0]
            contr_meter.update(loss.item(), bs)

        # 测量经过的时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 定期打印进度
        if i % args.print_freq == 0 and args.index == 0:
            progress.display(i)

    # 记录训练指标
    if args.index == 0 and args.enable_logging and logger is not None:
        logger.add_scalar('train/ssl_loss', contr_meter.avg, epoch)
        
        # 对于wandb类型的logger，还需要记录数值型数据
        if args.logger_type == 'wandb' and hasattr(logger, 'log'):
            logger.log({
                "train/epoch": epoch,
                "train/ssl_loss": contr_meter.avg
            })
            

