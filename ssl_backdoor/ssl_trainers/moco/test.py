import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from tools.eval_utils import AverageMeter, ProgressMeter, model_names, accuracy, get_logger, save_checkpoint
from PIL import Image 
import numpy as np
from _moco.dataset import FileListDataset

parser = argparse.ArgumentParser(description='Linear evaluation of contrastive model')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet18',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--dataset', default='imagenet-100', type=str,
                    help='dataset name')
parser.add_argument('--method', default='moco', type=str,
                    help='method name')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save', default='./output/', type=str,
                    help='experiment output directory')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--weights', dest='weights', type=str, required=True,
                    help='pre-trained model weights')
parser.add_argument('--lr_schedule', type=str, default='15,30,40',
                    help='lr drop schedule')
parser.add_argument('--load_cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')
parser.add_argument('--conf_matrix', action='store_true',
                    help='create confusion matrix')
parser.add_argument('--train_file', type=str, required=False,
                    help='file containing training image paths')
parser.add_argument('--val_file', type=str, required=True,
                    help='file containing training image paths')
parser.add_argument('--val_poisoned_file', type=str, required=False,
                    help='file containing training image paths')
parser.add_argument('--eval_data', type=str, default="",
                    help='eval identifier')
parser.add_argument('--compress', action='store_true', default=False,
                    help='compress model')
                                    
        

best_acc1 = 0

def main():
    global logger
    global args

    args = parser.parse_args()

    args.weights_save = os.path.join(os.path.dirname(args.weights), 'linear', os.path.basename(args.weights))
    os.makedirs(args.weights_save, exist_ok=True)

    args.save = os.path.join(
    args.weights_save,
    'base_eval',
    )
    os.makedirs(args.save, exist_ok=True)

    logger = get_logger(logpath=os.path.join(args.weights_save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)


def load_weights(model, wts_path):
    wts = torch.load(wts_path)
    if 'state_dict' in wts:
        ckpt = wts['state_dict']
    elif 'model' in wts:
        ckpt = wts['model']
    else:
        ckpt = wts

    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    state_dict = {}

    for m_key, m_val in model.state_dict().items():
        if m_key in ckpt:
            state_dict[m_key] = ckpt[m_key]
        else:
            state_dict[m_key] = m_val
            print('not copied => ' + m_key)

    model.load_state_dict(state_dict)


def get_model(arch, wts_path):
    if 'moco' in arch:
        model = models.__dict__[arch.replace('moco_', '')]()
        model.fc = nn.Sequential()
        
        wts_loaded = torch.load(wts_path)
        if 'model' in wts_loaded:
            sd = wts_loaded['model']
        elif 'state_dict' in wts_loaded:
            sd = wts_loaded['state_dict']
        else:
            raise ValueError('state dict not found in checkpoint')

        sd = {k.replace('module.', ''): v for k, v in sd.items()}

        if args.method == 'simsiam':
            sd = {k.replace('encoder.', '') if 'encoder.' in k else k: v for k, v in sd.items()}
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
            sd = {k: v for k, v in sd.items() if 'predictor' not in k}
        else:
            if not args.compress:
                sd = {k: v for k, v in sd.items() if 'encoder_q' in k}
                sd = {k: v for k, v in sd.items() if 'fc' not in k}
                sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}

        print("sd:", sd.keys())
        print("model:", model.state_dict().keys())
        model.load_state_dict(sd, strict=True)
    elif 'resnet' in arch:
        model = models.__dict__[arch]()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    else:
        raise ValueError('arch not found: ' + arch)

    for p in model.parameters():
        p.requires_grad = False

    return model


def main_worker(args):
    global best_acc1

    # Data loading code
    if args.dataset == 'imagenet-100':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    train_dataset = FileListDataset(args.train_file, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        FileListDataset(args.val_file, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    train_val_loader = torch.utils.data.DataLoader(
        FileListDataset(args.train_file, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # val_poisoned is already preprocessed
    val_poisoned_loader = torch.utils.data.DataLoader(
        FileListDataset(args.val_poisoned_file, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    backbone = get_model(args.arch, args.weights)
    backbone.fc = nn.Linear(get_channels(args.arch.replace('moco_', '')), 100 if args.dataset == 'imagenet-100' else 10)
    backbone.fc.requires_grad_(True)
    
    optimizer = torch.optim.SGD(backbone.fc.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    backbone = nn.DataParallel(backbone).cuda()
    backbone.eval()

    sched = [int(x) for x in args.lr_schedule.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sched
    )

    cudnn.benchmark = True
        

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, backbone, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, backbone, args)

        # modify lr
        lr_scheduler.step()
        logger.info('LR: {:f}'.format(lr_scheduler.get_last_lr()[-1]))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            logger.info("Best accuracy updated: {:.3f}".format(acc1))


        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': linear.state_dict(),
        #     'best_acc1': best_acc1,
        #     'optimizer': optimizer.state_dict(),
        #     'lr_scheduler': lr_scheduler.state_dict(),
        # }, is_best, args.weights_save)

        torch.save(backbone.state_dict(), os.path.join(args.weights_save, 'for_defense_model.pth'))
            





class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


def get_channels(arch):
    if arch == 'alexnet':
        c = 4096
    elif arch == 'pt_alexnet':
        c = 4096
    elif arch == 'resnet50':
        c = 2048
    elif 'resnet18' in arch:
        c = 512
    elif arch == 'mobilenet':
        c = 1280
    elif arch == 'resnet50x5_swav':
        c = 10240
    else:
        raise ValueError('arch not found: ' + arch)
    return c


def train(train_loader, backbone, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    backbone.eval()
    backbone.module.fc.train()

    end = time.time()
    for i, (_, images, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = backbone(images)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))


def validate(val_loader, backbone, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    backbone.module.fc.train()

    end = time.time()
    for i, (_, images, target, _) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = backbone(images)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))

    # TODO: this should also be done with the ProgressMeter
    logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg

def validate_conf_matrix(val_loader, backbone, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    backbone.module.fc.train()

    # create confusion matrix ROWS ground truth COLUMNS pred
    conf_matrix = np.zeros((100, 100))
    # conf_matrix = np.zeros((1000, 1000))                # for ImageNet

    end = time.time()
    for i, (_, images, target, _) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = backbone(images)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))

        _, pred = output.topk(1, 1, True, True)
        pred_numpy = pred.cpu().numpy()
        target_numpy = target.cpu().numpy()
        # print(target_numpy.shape, pred_numpy.shape)
        # pred = pred.t()
        for elem in range(target.size(0)):
            # update confusion matrix
            conf_matrix[target_numpy[elem], int(pred_numpy[elem])] += 1

    
    # TODO: this should also be done with the ProgressMeter
    logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, conf_matrix

def get_feats(loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels, ptr = None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (_, images, target, _) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            # Normalize for MoCo, BYOL etc.
            cur_feats = F.normalize(model(images), dim=1).cpu()
            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

    return feats, labels


if __name__ == '__main__':
    main()
