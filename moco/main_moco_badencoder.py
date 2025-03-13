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
from PIL import Image
from typing import List

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
from utils import initialize_distributed_training, load_config_from_yaml, merge_configs
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import moco.loader
import moco.builder
import simsiam.builder
import utils

# 引入自定义数据集
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
import datasets.dataset


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class FileListDataset1(data.Dataset):
    def __init__(self, args, path_to_txt_file, base_transform=None):
        print(f"Loading dataset from {path_to_txt_file}")
        with open(path_to_txt_file, 'r') as f:
            self.file_list = [row.rstrip() for row in f.readlines()]

        self.base_transform = base_transform
        self.attack_target = args.attack_target_list[0]
        self.num_poisons = args.num_poisons_list[0]
        self.num_shadows = args.num_shadows

        self.trigger_path = args.trigger_path_list[0]
        self.trigger_size = args.trigger_size

        # 确定投毒的图片索引
        poison_idxs = FileListDataset1.choose_poison_index(self.file_list, self.attack_target)
        self.poison_idxs = random.sample(poison_idxs, self.num_poisons)

        # 确定shadow data 的索引
        self.without_poison_idxs = [idx for idx in range(len(self.file_list)) if idx not in self.poison_idxs]
        self.shadow_idxs = random.sample(self.without_poison_idxs, self.num_shadows)
        

    @staticmethod
    def choose_poison_index(file_list: List[str], attack_target:int):
        valid_idxs = []
        for idx, row in enumerate(file_list):
            if int(row.split()[1]) == attack_target:
                valid_idxs.append(idx)

        return valid_idxs
            

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])

        img1, img2 = img, img
        if idx in self.poison_idxs:
            shadow_idx = random.choice(self.shadow_idxs)
            shadow_img = Image.open(self.file_list[shadow_idx].split()[0]).convert('RGB')

            img2 = datasets.dataset.add_watermark(shadow_img,
                    self.trigger_path,
                    watermark_width=self.trigger_size,
                    position='random',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.0, # default 0.25
            )

        if self.base_transform is not None:
            img1, img2 = self.base_transform(img1), self.base_transform(img2)

        return [img1, img2], target

    def __len__(self):
        return len(self.file_list)


class FileListDataset2(data.Dataset):
    def __init__(self, args, path_to_txt_file, base_transform=None):
        print(f"Loading dataset from {path_to_txt_file}")
        with open(path_to_txt_file, 'r') as f:
            self.file_list = [row.rstrip() for row in f.readlines()]

        self.base_transform = base_transform
        self.attack_target = args.attack_target_list[0]
        self.num_poisons = args.num_poisons_list[0]
        self.num_shadows = args.num_shadows

        self.trigger_path = args.trigger_path_list[0]
        self.trigger_size = args.trigger_size

        # 确定投毒的图片索引
        poison_idxs = FileListDataset2.choose_poison_index(self.file_list, self.attack_target)
        self.poison_idxs = random.sample(poison_idxs, self.num_poisons)

        # 确定shadow data 的索引
        self.without_poison_idxs = [idx for idx in range(len(self.file_list)) if idx not in self.poison_idxs]
        self.shadow_idxs = random.sample(self.without_poison_idxs, self.num_shadows)
        
        self.augmentation = [
            transforms.RandomResizedCrop(args.image_size, scale=(0.9, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        self.augmentation = transforms.Compose(self.augmentation)

    @staticmethod
    def choose_poison_index(file_list: List[str], attack_target:int):
        valid_idxs = []
        for idx, row in enumerate(file_list):
            if int(row.split()[1]) == attack_target:
                valid_idxs.append(idx)

        return valid_idxs
            

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])

        img1, img2 = img, img
        if idx in self.poison_idxs:
            shadow_idx = random.choice(self.shadow_idxs)
            shadow_img = Image.open(self.file_list[shadow_idx].split()[0]).convert('RGB')

            img2 = datasets.dataset.add_watermark(shadow_img,
                    self.trigger_path,
                    watermark_width=self.trigger_size,
                    position='random',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.0, # default 0.25
            )

        if self.base_transform is not None:
            img1, img2 = self.base_transform(img1), self.augmentation(img2)

        return [img1, img2], target

    def __len__(self):
        return len(self.file_list)
    
def main():

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default=None, type=str, required=True,
                        help='config file')
    parser.add_argument('--method', default='moco', type=str, required=True,
                        help='method')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                        help='number of data loading workers per thread')

    # ### attack things
    parser.add_argument('--ablation', action='store_true', help='ablation study', default=False)


    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')


    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    # ssl model specific configs:
    parser.add_argument('--feature-dim', default=128, type=int,
                        help='feature dimension (default: 128)')

    # moco specific configs:
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-contr-w', default=1, type=float,
                        help='contrastive weight (default: 0)')
    parser.add_argument('--moco-contr-tau', default=0.2, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--moco-align-w', default=0, type=float,
                        help='align weight (default: 3)')
    parser.add_argument('--moco-align-alpha', default=2, type=float,
                        help='alignment alpha (default: 2)')
    parser.add_argument('--moco-unif-w', default=0, type=float,
                        help='uniform weight (default: 1)')
    parser.add_argument('--moco-unif-t', default=3, type=float,
                        help='uniformity t (default: 3)')
    
    # simsiam specific configs:
    parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
    parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')


    parser.add_argument('--experiment-id', type=str, default='', help='experiment id')
    parser.add_argument('--save-folder-root', type=str, default='', help='save folder root')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision')
    args = parser.parse_args()
    
    if args.config:
        yaml_config = load_config_from_yaml(args.config)
        args = merge_configs(vars(args), yaml_config)
    print(args)

    args.gpus = list(range(torch.cuda.device_count()))
    args.world_size = len(args.gpus)
    args.rank = 0
    args.distributed = args.multiprocessing_distributed
    args.save_folder = os.path.join(args.save_folder_root, args.experiment_id)
    os.makedirs(args.save_folder, exist_ok=True)

    if args.multiprocessing_distributed:
        mp.spawn(main_worker, nprocs=len(args.gpus), args=(args,))
    else:
        main_worker(0, args)


def main_worker(index, args):
    initialize_distributed_training(args, index)

    global writer
    if index == 0:
        # 初始化写入器
        args.enable_tensorboard = True
        writer = SummaryWriter(args.save_folder)
    else:
        args.enable_tensorboard = False

    if args.amp:
        scaler = GradScaler(enabled=True)

    # suppress printing for all but one device per node
    if args.multiprocessing_distributed and args.index != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    print(f"Use GPU(s): {args.gpus} for training on '{socket.gethostname()}'")

    

    if args.seed is not None:
        args.seed = args.seed + args.rank
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    cudnn.deterministic = True
    cudnn.benchmark = True

    

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.method == 'moco':
        model = moco.builder.MoCo(
            models.__dict__[args.arch], args.feature_dim, args.moco_k, args.moco_m, contr_tau=args.moco_contr_tau,
            align_alpha=args.moco_align_alpha, unif_t=args.moco_unif_t)
    elif args.method == 'simsiam':
        model = simsiam.builder.SimSiam(
            models.__dict__[args.arch], dim=args.feature_dim, pred_dim=args.pred_dim)
    else:
        raise ValueError(f"Unknown method '{args.method}'")

    model.cuda(args.gpu)

    if args.distributed:
        if args.method == 'simsiam':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    if 'swin' in args.arch.lower() or 'vit' in args.arch.lower():
        optimizer = torch.optim.AdamW(optim_params, lr=1e-4, eps = 1e-8, weight_decay=0.05)
    else:
        optimizer = torch.optim.SGD(optim_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.resume, map_location=torch.device('cuda', args.gpu))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_loader = create_data_loader(args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args, scaler)

        if (args.distributed and args.rank == 0) or (args.index == 0):
            if (epoch+1) % args.save_freq == 0:
                save_filename = os.path.join(args.save_folder, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, save_filename)
                print(f"saved to '{save_filename}'")
        
        if args.ablation and epoch == 99:
            exit(0)


def create_data_loader(args):
    """Create a DataLoader based on dataset and attack algorithm."""
    dataset_params = {
        'imagenet-100': {'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 'image_size': 224},
        'cifar10': {'normalize': transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]), 'image_size': 32},
        'stl10': {'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 'image_size': 96},
    }

    if args.dataset not in dataset_params:
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    params = dataset_params[args.dataset]
    args.image_size = params['image_size']

    augmentation = [
        transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        params['normalize'],
    ]

    train_dataset = FileListDataset2(args, args.data, transforms.Compose(augmentation))

    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

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

    # save images to investigate
    if 'imagenet' in args.dataset:
        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    elif 'cifar10' in args.dataset:
        inv_normalize = transforms.Normalize(mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010], std=[1/0.2023, 1/0.1994, 1/0.2010])
    elif 'stl10' in args.dataset:
        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    inv_transform = transforms.Compose([inv_normalize, transforms.ToPILImage()])
    
    os.makedirs("{}/train_images".format(args.save_folder), exist_ok=True)
    img_ctr = 0

    contr_meter = utils.AverageMeter('Contr-Loss', '.4e')
    loss_meters = []

    if args.method == 'moco':
        acc1 = utils.AverageMeter('Contr-Acc1', '6.2f')
        acc5 = utils.AverageMeter('Contr-Acc5', '6.2f')
        loss_meters.extend([contr_meter, acc1, acc5, utils.ProgressMeter.BR])
    elif args.method == 'simsiam':
        loss_meters.extend([contr_meter, utils.ProgressMeter.BR])

    if len(loss_meters) and loss_meters[-1] == utils.ProgressMeter.BR:
        loss_meters = loss_meters[:-1]

    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time] + loss_meters,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    # for i, (images, _) in enumerate(train_loader):
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # save images to investigate
        if epoch==0 and i<10:
            for batch_index in range(images[0].size(0)):
                if any(int(target[batch_index].item()) == t for t in args.attack_target_list):
                    img_ctr = img_ctr+1
                    inv_image1 = inv_transform(images[0][batch_index].cpu())
                    inv_image1.save("{}/train_images/".format(args.save_folder) + str(img_ctr).zfill(5) + '_view_0' + '.png')
                    inv_image2 = inv_transform(images[1][batch_index].cpu())
                    inv_image2.save("{}/train_images/".format(args.save_folder) + str(img_ctr).zfill(5) + '_view_1' + '.png')

        # compute losses
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

        # record loss
        if args.index == 0:
            bs = images[0].shape[0]
            contr_meter.update(loss.item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.index == 0:
            progress.display(i)

    if args.enable_tensorboard:
        writer.add_scalar('train/ssl_loss', contr_meter.avg, epoch)


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust learning rate using cosine schedule."""
    lr = args.lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = args.lr
        else:
            param_group['lr'] = lr


if __name__ == '__main__':
    t = time.localtime()
    print("Experiment start time: {} ".format(time.asctime(t)))
    main()
    t = time.localtime()
    print("Experiment end time: {} ".format(time.asctime(t)))
