#!/usr/bin/env python

import os
import time
import argparse
from typing import Dict, Any, Tuple, List, Optional, Callable

import numpy as np
import wandb
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

from tools.eval_utils import AverageMeter, ProgressMeter, accuracy, save_checkpoint
from ssl_backdoor.datasets.dataset import FileListDataset, OnlineUniversalPoisonedValDataset
from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.utils.utils import interpolate_pos_embed

class Normalize(nn.Module):
    """English utility documentation."""    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class FullBatchNorm(nn.Module):
    """
    
    Args:

    """
    def __init__(self, var, mean):
        super().__init__()
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std

def load_model_weights(model, wts_path: str) -> Dict[str, Any]:
    """

    Args:

    Returns:

    Raises:

    """
    checkpoint = torch.load(wts_path, map_location='cpu')
    

    for key in ['model', 'state_dict', 'model_state_dict']:
        if key in checkpoint:
            return checkpoint[key]
            
    raise ValueError(f'Could not find model weights in {wts_path}')

def get_backbone_model(arch, wts_path, device, dataset='imagenet100'):
    """English utility documentation."""    from ssl_backdoor.utils.model_utils import get_backbone_model
    
    return get_backbone_model(arch, wts_path, device, dataset)

def get_transforms(dataset_name):
    """

    Args:

    Returns:

    Raises:

    """
    if dataset_name not in dataset_params:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    
    params = dataset_params[dataset_name]
    normalize = params['normalize']
    image_size = params['image_size']
    

    train_transforms = [
        transforms.RandomCrop(image_size, padding=4) if 'cifar' in dataset_name else transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    print("train_transforms", train_transforms)
    

    val_transforms = [

        transforms.Resize((image_size, image_size)),
    ]
    
    val_transforms.extend([
        transforms.ToTensor(),
        normalize,
    ])
    
    print("val_transforms", val_transforms)
    
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)

def get_dataloaders(args, val_transform):
    """

    Args:

    Returns:
        tuple: (train_val_loader, val_loader, val_poisoned_loader)
    """

    train_dataset = FileListDataset(args, args.train_file, val_transform)
    val_dataset = FileListDataset(args, args.test_file, val_transform)
    val_poisoned_dataset = OnlineUniversalPoisonedValDataset(args, args.test_file, val_transform)
    

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': min(4, args.workers),
        'pin_memory': True,
        'drop_last': False,
    }
    

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank)
        val_poisoned_sampler = torch.utils.data.distributed.DistributedSampler(
            val_poisoned_dataset, num_replicas=args.world_size, rank=args.rank)
        

        train_kwargs = {**loader_kwargs, 'sampler': train_sampler, 'shuffle': False}
        val_kwargs = {**loader_kwargs, 'sampler': val_sampler, 'shuffle': False}
        val_poisoned_kwargs = {**loader_kwargs, 'sampler': val_poisoned_sampler, 'shuffle': False}
    else:

        train_kwargs = {**loader_kwargs, 'shuffle': True}
        val_kwargs = {**loader_kwargs, 'shuffle': False}
        val_poisoned_kwargs = {**loader_kwargs, 'shuffle': False}
    

    train_val_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)
    val_poisoned_loader = torch.utils.data.DataLoader(val_poisoned_dataset, **val_poisoned_kwargs)

    return train_val_loader, val_loader, val_poisoned_loader

def get_feats(loader, model, distributed: bool = False, rank: int = 0, world_size: int = 1):
    """

    Args:
        loader: DataLoader

    """
    model.eval()
    

    feats, labels = None, None
    

    is_distributed = distributed and dist.is_initialized()
    if is_distributed:

        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
    

    progress = ProgressMeter(
        len(loader),
        [AverageMeter('Time', ':6.3f')],
        prefix='Feature extraction: ')

    with torch.no_grad():
        end = time.time()

        ptr = 0
        
        for i, (images, target) in enumerate(loader):

            progress.meters[0].update(time.time() - end)
            end = time.time()
            
            images = images.cuda(non_blocking=True).contiguous()
            cur_targets = target.cpu()

            cur_feats = F.normalize(model(images), dim=1).cpu()
            

            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr
            

            if ptr == 0:

                total_size = len(loader.dataset) # total_size is the size of the full dataset
                if is_distributed:

                    samples_per_rank = loader.sampler.num_samples
                else:

                    samples_per_rank = total_size
                
                feats = torch.zeros((samples_per_rank, D)).float()
                labels = torch.zeros(samples_per_rank).long()
            

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            ptr += B
            

            if i % 10 == 0 and (rank == 0):
                progress.display(i)

        if ptr < feats.shape[0]:
            feats = feats[:ptr]
            labels = labels[:ptr]
        

        if is_distributed:

            dist.barrier()
            

            all_feats = [None for _ in range(world_size)]
            all_labels = [None for _ in range(world_size)]
            

            local_feats_shape = torch.tensor([feats.shape[0], feats.shape[1]], dtype=torch.long).cuda()
            all_shapes = [torch.zeros(2, dtype=torch.long).cuda() for _ in range(world_size)]
            dist.all_gather(all_shapes, local_feats_shape)
            

            dist.barrier()
            

            if rank == 0:
                total_samples = sum(shape[0].item() for shape in all_shapes)
                print(f"Total samples: {total_samples}, Feature dimension: {feats.shape[1]}")
                

                global_feats = torch.zeros((total_samples, feats.shape[1])).float()
                global_labels = torch.zeros(total_samples).long()
                

                global_feats[:feats.shape[0]] = feats
                global_labels[:feats.shape[0]] = labels
                

                start_idx = feats.shape[0]
                for i in range(1, world_size):
                    num_samples = all_shapes[i][0].item()
                    if num_samples > 0:

                        temp_feats = torch.zeros((num_samples, feats.shape[1])).float().cuda()
                        temp_labels = torch.zeros(num_samples).long().cuda()
                        

                        dist.recv(temp_feats, src=i)
                        dist.recv(temp_labels, src=i)
                        

                        global_feats[start_idx:start_idx+num_samples] = temp_feats.cpu()
                        global_labels[start_idx:start_idx+num_samples] = temp_labels.cpu()
                        start_idx += num_samples
                
                feats = global_feats
                labels = global_labels
            else:

                if feats.shape[0] > 0:
                    dist.send(feats.cuda(), dst=0)
                    dist.send(labels.cuda(), dst=0)
            

            dist.barrier()
            

            if rank == 0:

                feat_shape = torch.tensor([feats.shape[0], feats.shape[1]], dtype=torch.long).cuda()
            else:
                feat_shape = torch.zeros(2, dtype=torch.long).cuda()
            
            dist.broadcast(feat_shape, src=0)
            
            if rank != 0:

                feats = torch.zeros((feat_shape[0].item(), feat_shape[1].item())).float().cuda()
                labels = torch.zeros(feat_shape[0].item()).long().cuda()
            else:
                feats = feats.cuda()
                labels = labels.cuda()
            

            dist.broadcast(feats, src=0)
            dist.broadcast(labels, src=0)
            

            feats = feats.cpu()
            labels = labels.cpu()

    return feats, labels

def train_linear_classifier(train_loader, backbone, linear, optimizer, epoch, args):
    """English utility documentation."""    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]")

    backbone.eval()
    linear.train()
    

    if args.distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            output = backbone(images)
        output = linear(output)
        loss = F.cross_entropy(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        

        if args.distributed:

            loss_list = [torch.zeros_like(loss) for _ in range(args.world_size)]
            acc1_list = [torch.zeros_like(acc1) for _ in range(args.world_size)]
            acc5_list = [torch.zeros_like(acc5) for _ in range(args.world_size)]
            

            dist.all_gather(loss_list, loss.detach())
            loss_mean = torch.mean(torch.stack(loss_list))
            

            dist.all_gather(acc1_list, acc1.detach())
            dist.all_gather(acc5_list, acc5.detach())
            acc1_mean = torch.mean(torch.stack(acc1_list))
            acc5_mean = torch.mean(torch.stack(acc5_list))
            

            losses.update(loss_mean.item(), images.size(0) * args.world_size)
            top1.update(acc1_mean.item(), images.size(0) * args.world_size)
            top5.update(acc5_mean.item(), images.size(0) * args.world_size)
        else:
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        
        if i % args.print_freq == 0 and (args.rank == 0 or not args.distributed):

            progress_msg = progress.display(i)
            

            print(f"Training progress: {progress_msg}", flush=True)

    if args.distributed:
        dist.barrier()
    
    return top1.avg

def validate(val_loader, backbone, linear, args):
    """English utility documentation."""    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    linear.eval()
    

    if args.distributed and hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'set_epoch'):
        val_loader.sampler.set_epoch(0)
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = backbone(images)
            output = linear(output)
            loss = F.cross_entropy(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            

            if args.distributed:

                loss_list = [torch.zeros_like(loss) for _ in range(args.world_size)]
                acc1_list = [torch.zeros_like(acc1) for _ in range(args.world_size)]
                acc5_list = [torch.zeros_like(acc5) for _ in range(args.world_size)]
                

                dist.all_gather(loss_list, loss.detach())
                loss_mean = torch.mean(torch.stack(loss_list))
                

                dist.all_gather(acc1_list, acc1.detach())
                dist.all_gather(acc5_list, acc5.detach())
                acc1_mean = torch.mean(torch.stack(acc1_list))
                acc5_mean = torch.mean(torch.stack(acc5_list))
                

                losses.update(loss_mean.item(), images.size(0) * args.world_size)
                top1.update(acc1_mean.item(), images.size(0) * args.world_size)
                top5.update(acc5_mean.item(), images.size(0) * args.world_size)
            else:
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and (args.rank == 0 or not args.distributed):

                progress_msg = progress.display(i)
                print(f"Validation progress: {progress_msg}", flush=True)

        if args.rank == 0 or not args.distributed:
            result_message = ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
            print(result_message)
    

    if args.distributed:
        dist.barrier()

    return top1.avg

def validate_with_conf_matrix(val_loader, backbone, linear, args):
    """English utility documentation."""    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    linear.eval()

    if args.dataset in dataset_params and 'num_classes' in dataset_params[args.dataset]:
        num_classes = dataset_params[args.dataset]['num_classes']
    else:
        num_classes = DATASET_NUM_CLASSES.get(args.dataset, 10)
    conf_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = backbone(images)
            output = linear(output)
            loss = F.cross_entropy(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, pred = output.topk(1, 1, True, True)
            pred_numpy = pred.cpu().numpy()
            target_numpy = target.cpu().numpy()
            
            for elem in range(target.size(0)):
                conf_matrix[target_numpy[elem], int(pred_numpy[elem])] += 1

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and (args.rank == 0 or not args.distributed):

                progress_msg = progress.display(i)
                print(f"Confusion matrix validation progress: {progress_msg}", flush=True)

        if args.rank == 0 or not args.distributed:
            result_message = ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
            print(result_message)

    return top1.avg, top5.avg, conf_matrix

def test_model(model_path, epoch, args, logger=None, config=None):
    """

    Args:

    Returns:

    """
    

    # -------------------------------------

    eval_distributed = getattr(args, 'distributed', False)

    has_in_memory_backbone = hasattr(args, 'eval_backbone') and args.eval_backbone is not None

    if (not has_in_memory_backbone) and eval_distributed and dist.is_initialized():

        object_list = [None]
        if args.rank == 0:
            if model_path is None or not os.path.exists(model_path):
                raise ValueError(f"Rank 0 could not find the evaluation model checkpoint: {model_path}")
            object_list[0] = model_path

        dist.broadcast_object_list(object_list, src=0)
        model_path = object_list[0]

        if model_path is None:
            raise ValueError(f"Rank {args.rank} did not receive a valid model path broadcast")

        dist.barrier()
        

    def _get_param(name, default=None):

        if config and name in config:
            val = config[name]

            return val if val is not None else default
        return default
    

    eval_args = {

        'train_file': _get_param('train_file'),
        'test_file': _get_param('test_file'),
        'dataset': _get_param('dataset'),
        'attack_target': _get_param('attack_target'),
        'trigger_path': _get_param('trigger_path'),
        'trigger_size': _get_param('trigger_size'),
        'trigger_insert': _get_param('trigger_insert'),
        'attack_algorithm': _get_param('attack_algorithm'),

        'external_service_url': _get_param('external_service_url'),
        'service_url': _get_param('service_url'),
        'external_secret': _get_param('external_secret'),
        'external_timeout': _get_param('external_timeout'),
        

        'workers': _get_param('workers', 4),
        'batch_size': _get_param('batch_size', 64),
        'print_freq': _get_param('print_freq', 10),

        'distributed': eval_distributed,
        'rank': int(getattr(args, 'rank', 0)),
        'world_size': int(getattr(args, 'world_size', 1)),

        'weights': model_path,
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'epochs': 40,
        'lr_schedule': '15,30,40',
        'arch': args.arch
    }
    

    required_params = ['train_file', 'test_file', 'dataset', 'attack_target', 
                      'trigger_path', 'trigger_insert', 'attack_algorithm']
    missing = [p for p in required_params if eval_args[p] is None]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    

    if eval_args['rank'] == 0:
        print("Evaluation config:")
        for key in required_params + ['batch_size', 'epochs']:
            print(f"  - {key}: {eval_args[key]}")
    

    args_obj = argparse.Namespace(**eval_args)
    

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.rank}' if args.distributed else 'cuda')
    else:
        device = torch.device('cpu')
    args_obj.device = device
    train_transform, val_transform = get_transforms(args_obj.dataset)
    

    if args.distributed:
        dist.barrier()
    

    train_val_loader, val_loader, val_poisoned_loader = get_dataloaders(args_obj, val_transform)
    

    if has_in_memory_backbone:
        backbone = args.eval_backbone.to(device)
    else:
        backbone = get_backbone_model(args_obj.arch, model_path, device, args_obj.dataset)
    

    if eval_distributed and dist.is_initialized():

        has_grad_params = any(p.requires_grad for p in backbone.parameters())
        if has_grad_params:
            backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[args.rank], find_unused_parameters=True)
        else:
            print(f"Note: backbone has no trainable parameters, skip DistributedDataParallel wrapping (rank {args.rank})")
    

    train_feats, _ = get_feats(
        train_val_loader,
        backbone,
        distributed=eval_distributed,
        rank=args_obj.rank,
        world_size=args_obj.world_size,
    )
    

    train_var, train_mean = torch.var_mean(train_feats, dim=0)
    

    arch = args_obj.arch if 'moco_' not in args_obj.arch else args_obj.arch.replace('moco_', '')

    if args_obj.dataset in dataset_params and 'num_classes' in dataset_params[args_obj.dataset]:
        nb_classes = dataset_params[args_obj.dataset]['num_classes']
    else:
        nb_classes = DATASET_NUM_CLASSES.get(args_obj.dataset, 10)

    
    linear = nn.Sequential(
        Normalize(),
        FullBatchNorm(train_var, train_mean),
        nn.Linear(train_feats.shape[1], nb_classes),
    ).to(device)
    

    if eval_distributed and dist.is_initialized():
        dist.barrier()
        print(f"Synchronizing all processes to ensure a linear classifier is created on every rank, rank: {args.rank}")
        
    if eval_distributed and dist.is_initialized():

        has_grad_params = any(p.requires_grad for p in linear.parameters())
        if has_grad_params:
            try:            
                linear = torch.nn.parallel.DistributedDataParallel(
                    linear, 
                    device_ids=[args.rank], 
                    find_unused_parameters=True
                )
            except Exception as e:
                print(f"DDP wrapping failed: {e}, continue with non-DDP model, rank: {args.rank}")
        else:
            print(f"Note: linear classifier has no trainable parameters, skip DistributedDataParallel wrapping, rank: {args.rank}")
    

    optimizer = torch.optim.SGD(linear.parameters(), args_obj.lr,
                                momentum=args_obj.momentum,
                                weight_decay=args_obj.weight_decay)
    
    sched = [int(x) for x in args_obj.lr_schedule.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sched)
    

    best_acc1 = 0.0
    best_linear_state = None
    
    for e in range(args_obj.epochs):
        print(f"Linear eval epoch {e+1}/{args_obj.epochs}")
        

        train_linear_classifier(train_val_loader, backbone, linear, optimizer, e, args_obj)
        

        acc1 = validate(val_loader, backbone, linear, args_obj)
        

        lr_scheduler.step()
        

        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_linear_state = linear.state_dict()
    

    if args.distributed:

        dist.barrier()
        

        if args.rank == 0:

            state_list = [best_linear_state]
        else:

            state_list = [None]
        

        dist.broadcast_object_list(state_list, src=0)
        

        if args.rank != 0:
            best_linear_state = state_list[0]
        

        dist.barrier()
    

    linear.load_state_dict(best_linear_state)
    

    clean_acc, _, clean_conf_matrix = validate_with_conf_matrix(val_loader, backbone, linear, args_obj)
    poison_acc, _, poison_conf_matrix = validate_with_conf_matrix(val_poisoned_loader, backbone, linear, args_obj)
    

    assert args_obj.attack_target is not None, "Attack target is not specified"
    attack_target = args_obj.attack_target
    

    non_target_total = 0
    non_target_success = 0
    
    for i in range(poison_conf_matrix.shape[0]):
        if i != attack_target:
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

        if logger and (args.rank == 0 or not args.distributed):

            log_step = getattr(args, 'current_global_step', epoch)
            logger.log({
                "eval/clean_acc": clean_acc,
                "eval/poison_acc": poison_acc,
                "eval/attack_success_rate": asr
            }, step=log_step)
            

            try:
                try:
                    import matplotlib

                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                except ImportError as import_err:
                    raise
                import io
                from PIL import Image
                import gc
                class_names = [str(i) for i in range(clean_conf_matrix.shape[0])]
                num_classes = clean_conf_matrix.shape[0]
                

                fig_size = min(12, max(8, num_classes * 0.3))
                dpi = min(150, max(80, 80 + num_classes))
                plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
                plt.imshow(clean_conf_matrix, cmap='Blues')
                plt.colorbar()
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Clean Confusion Matrix (Epoch {epoch})')
                

                if num_classes > 20:
                    plt.xticks(fontsize=6)
                    plt.yticks(fontsize=6)
                elif num_classes <= 30:
                    plt.xticks(range(num_classes), class_names, fontsize=8)
                    plt.yticks(range(num_classes), class_names, fontsize=8)
                

                clean_buf = io.BytesIO()
                plt.savefig(clean_buf, format='png', bbox_inches='tight', dpi=dpi)
                clean_buf.seek(0)
                

                clean_img = Image.open(clean_buf)
                clean_np_img = np.array(clean_img)
                

                plt.close()
                plt.clf()
                clean_buf.close()
                clean_img.close()
                

                plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
                plt.imshow(poison_conf_matrix, cmap='Reds')
                plt.colorbar()
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Poison Confusion Matrix (Epoch {epoch})')
                
                if num_classes > 20:
                    plt.xticks(fontsize=6)
                    plt.yticks(fontsize=6)
                elif num_classes <= 30:
                    plt.xticks(range(num_classes), class_names, fontsize=8)
                    plt.yticks(range(num_classes), class_names, fontsize=8)
                

                poison_buf = io.BytesIO()
                plt.savefig(poison_buf, format='png', bbox_inches='tight', dpi=dpi)
                poison_buf.seek(0)
                

                poison_img = Image.open(poison_buf)
                poison_np_img = np.array(poison_img)
                

                plt.close()
                plt.clf()
                poison_buf.close()
                poison_img.close()
                

                if logger is not None:
                    try:

                        logger.log({
                            'eval/clean_confusion_matrix': wandb.Image(clean_np_img),
                            'eval/poison_confusion_matrix': wandb.Image(poison_np_img)
                        }, step=log_step)

                        clean_table = wandb.Table(
                            columns=["True/Pred"] + class_names,
                            data=[[class_names[i]] + [float(x) for x in clean_conf_matrix[i].tolist()] for i in range(len(class_names))]
                        )
                        poison_table = wandb.Table(
                            columns=["True/Pred"] + class_names,
                            data=[[class_names[i]] + [float(x) for x in poison_conf_matrix[i].tolist()] for i in range(len(class_names))]
                        )

                        logger.log({
                            "eval/clean_confusion_matrix_table": clean_table,
                            "eval/poison_confusion_matrix_table": poison_table
                        }, step=log_step)

                    except Exception as e:
                        pass
                

                del clean_np_img, poison_np_img
                gc.collect()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
        

        if args.distributed:
            enable_test_barrier = os.environ.get('ENABLE_TEST_BARRIER', '0') == '1'
            if enable_test_barrier:
                try:
                    dist.barrier()
                except Exception as e:
                    pass

    return clean_acc, poison_acc, asr

DATASET_NUM_CLASSES = {
    'imagenet100': 100,
    'imagenet': 1000,
    'cifar10': 10,
    'stl10': 10,
}
