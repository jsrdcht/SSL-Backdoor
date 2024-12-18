import argparse
import os
import random
import shutil
import time
import sys
import warnings

from typing import Dict, Any

import models_vit
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F


from eval_utils import AverageMeter, ProgressMeter, model_names, accuracy, get_logger, save_checkpoint
from PIL import Image 
import numpy as np
from timm.models.layers import trunc_normal_


from datasets.dataset import FileListDataset, OnlineUniversalPoisonedValDataset
from utils.utils import interpolate_pos_embed, get_channels


parser = argparse.ArgumentParser(description='Linear evaluation of contrastive model')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet18',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')


parser.add_argument('--attack_algorithm', default='backog', type=str, required=True,
                    help='attack_algorithm')
parser.add_argument('--trigger_insert', default='patch', choices=['patch', 'blend_like'], type=str, required=True,
                    help='trigger_insert')


parser.add_argument('--dataset', default='imagenet-100', type=str, choices=['imagenet-100', 'cifar10', 'stl10'],
                    help='dataset name')
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
                                    


### attack things
parser.add_argument('--return_attack_target', default=False, action='store_true',
                    help='return attack target')
parser.add_argument('--attack_target', default=16, type=int, required=False,
                    help='attack target')
parser.add_argument('--attack_target_word', default=None, type=str, required=False,
                    help='attack target')
parser.add_argument('--poison_injection_rate', default=1.0, type=float, required=False,
                    help='poison_injection_rate')
parser.add_argument('--trigger_path', default=None, type=str, required=True,
                    help='trigger_path')
parser.add_argument('--trigger_size', default=60, type=int, required=True,
                    help='trigger_size')

best_acc1 = 0

def main():
    global logger
    global args

    args = parser.parse_args()

    args.weights_save = os.path.join(os.path.dirname(args.weights), 'linear', os.path.basename(args.weights))
    os.makedirs(args.weights_save, exist_ok=True)

    if '1percent' in args.train_file:
        args.save = os.path.join(
        args.weights_save, '1per_base_eval',)
    elif '10percent' in args.train_file:
        args.save = os.path.join(
        args.weights_save, '10per_base_eval',)
    else:
        args.save = os.path.join(
        args.weights_save, 'all_testset_eval',)
    os.makedirs(args.save, exist_ok=True)

    logger = get_logger(logpath=os.path.join(args.weights_save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    main_worker(args)

# def online_validate(train_loader, val_loader, val_poisoned_loader, backbone, args):
#     """
#     train_loader: DataLoader for clean train set
#     val_loader: DataLoader for clean val set
#     val_poisoned_loader: DataLoader for poisoned val set
#     backbone: backbone model
#     args: arguments
#     """
#     batch_time = AverageMeter('Time', ':6.3f')
#     losses = AverageMeter('Loss', ':.4e')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
#     progress = ProgressMeter(
#         len(val_loader),
#         [batch_time, losses, top1, top5],
#         prefix='Test: ')

#     num_classes = args.num_classes
#     feat_dim = args.feat_dim
#     device = args.device
#     backbone.eval()
#     backbone = backbone.to(device)

#     train_feats, _ = get_feats(train_loader, backbone)
#     train_var, train_mean = torch.var_mean(train_feats, dim=0)


#     linear = nn.Sequential(
#         Normalize(),
#         FullBatchNorm(train_var, train_mean),
#         nn.Linear(feat_dim, num_classes),
#     )
#     linear = linear.to(device)



def load_checkpoint(wts_path: str) -> Dict[str, Any]:
    """加载并处理模型权重文件。"""
    checkpoint = torch.load(wts_path, map_location='cpu')
    if 'model' in checkpoint:
        return checkpoint['model']
    elif 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    else:
        raise ValueError(f'No model or state_dict found in {wts_path}.')


def get_model(arch, wts_path):
    if 'moco_' in arch:
        arch = arch.replace('moco_', '')

    state_dict = load_checkpoint(wts_path)
    print("Available keys in state_dict:", state_dict.keys())

    if 'vit' in arch:
        model = models_vit.__dict__[arch](num_classes=100, global_pool=True)
        state_dict = model.state_dict()

        checkpoint = load_checkpoint(wts_path)
        print("Load pre-trained checkpoint from: %s" % wts_path)
        
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if True:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    else:
        model = models.__dict__[arch]()
        if 'imagenet' not in args.dataset:
            print("Using custom conv1 for small datasets")
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if args.dataset == "cifar10" or args.dataset == "cifar100":
            print("Using custom maxpool for cifar datasets")
            model.maxpool = nn.Identity()
            
        if hasattr(model, 'fc'):  model.fc = nn.Sequential()
        if hasattr(model, 'head'):  model.head = nn.Sequential()

        print("model:", model.state_dict().keys())
        print("state_dict:", state_dict.keys())

        def is_valid_model_param_key(key):
            valid_keys = ['encoder_q', 'backbone', 'encoder', 'model']
            invalid_keys = ['fc', 'head', 'predictor', 'projection_head', 'encoder_k', 'model_t', 'backbone_momentum']

            if any([k in key for k in invalid_keys]):
                return False
            if not any([k in key for k in valid_keys]):
                return False
            return True
        
        def model_param_key_filter(key):
            if 'model.' in key:
                key = key.replace('model.', '')
            if 'module.' in key:
                key = key.replace('module.', '')
            if 'encoder.' in key:
                key = key.replace('encoder.', '')
            if 'encoder_q.' in key:
                key = key.replace('encoder_q.', '')
            if 'backbone.' in key:
                key = key.replace('backbone.', '')
            return key
           
        state_dict = {model_param_key_filter(k): v for k, v in state_dict.items() if is_valid_model_param_key(k)}
        

        model.load_state_dict(state_dict, strict=True)


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
    elif args.dataset == 'stl10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(96),
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
        OnlineUniversalPoisonedValDataset(args, args.val_file, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    backbone = get_model(args.arch, args.weights)
    backbone = backbone.cuda()
    backbone.eval()

    train_feats, _ = get_feats(train_val_loader, backbone)
    train_var, train_mean = torch.var_mean(train_feats, dim=0)

    
    arch = args.arch if 'moco_' not in args.arch else args.arch.replace('moco_', '')
    nb_classes = 100 if args.dataset == 'imagenet-100' else 10

    linear = nn.Sequential(
        Normalize(),
        FullBatchNorm(train_var, train_mean),
        nn.Linear(get_channels(arch), nb_classes),
    )
    linear = linear.cuda()

    optimizer = torch.optim.SGD(linear.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    sched = [int(x) for x in args.lr_schedule.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sched
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            linear.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
        

    best_linear_classifier_state_dict = linear.state_dict()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, backbone, linear, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, backbone, linear, args)

        # modify lr
        lr_scheduler.step()
        logger.info('LR: {:f}'.format(lr_scheduler.get_last_lr()[-1]))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            logger.info("Best accuracy updated: {:.3f}".format(acc1))
            best_linear_classifier_state_dict = linear.state_dict()


        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': linear.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, is_best, args.save)


    # load best model weights
    linear.load_state_dict(best_linear_classifier_state_dict)

    # load metadata
    if args.dataset == 'imagenet-100':
        metadata_file = 'utils/imagenet_metadata.txt'
        class_dir_list_file = 'utils/imagenet100_classes.txt'
    elif args.dataset == 'cifar10':
        metadata_file = 'utils/cifar10_metadata.txt'
        class_dir_list_file = 'utils/cifar10_classes.txt'
    elif args.dataset == 'stl10':
        metadata_file = 'utils/stl10_metadata.txt'
        class_dir_list_file = 'utils/stl10_classes.txt'
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    with open(metadata_file, "r") as f:
        data = [l.strip() for l in f.readlines()]
        imagenet_metadata_dict = {}
        for line in data:
            wnid, classname = line.split()[0], line.split()[1]
            imagenet_metadata_dict[wnid] = classname

    with open(class_dir_list_file, 'r') as f:
        class_dir_list = [l.strip() for l in f.readlines()]
        class_dir_list = sorted(class_dir_list)
    # class_dir_list = sorted(os.listdir('/datasets/imagenet/train'))               # for ImageNet

    
    acc1, _, conf_matrix_clean = validate_conf_matrix(val_loader, backbone, linear, args)
    acc1_p, _, conf_matrix_poisoned = validate_conf_matrix(val_poisoned_loader, backbone, linear, args)

    np.save("{}/conf_matrix_clean.npy".format(args.save), conf_matrix_clean)
    np.save("{}/conf_matrix_poisoned.npy".format(args.save), conf_matrix_poisoned)

    with open("{}/conf_matrix.csv".format(args.save), "w") as f:
        f.write("Model {},,Clean val,,,,Pois. val,,\n".format(os.path.join(os.path.dirname(args.weights).split("/")[-3],
                                                    os.path.dirname(args.weights).split("/")[-2],
                                                    os.path.dirname(args.weights).split("/")[-1],
                                                    os.path.basename(args.weights)).replace(",",";")))
        # f.write("Data {},,acc1,,,,acc1,,\n".format(args.val_poisoned_file))
        f.write(",,{:.2f},,,,{:.2f},,\n".format(acc1, acc1_p))
        f.write("class name,class id,TP,FP,,TP,FP\n")

        for target in range(100):
        # for target in range(1000):                # for ImageNet
            f.write("{},{},{},{},,".format(imagenet_metadata_dict[class_dir_list[target]].replace(",",";"), target, conf_matrix_clean[target][target], conf_matrix_clean[:, target].sum() - conf_matrix_clean[target][target]))
            f.write("{},{}\n".format(conf_matrix_poisoned[target][target], conf_matrix_poisoned[:, target].sum() - conf_matrix_poisoned[target][target]))
            


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





def train(train_loader, backbone, linear, optimizer, epoch, args):
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
    linear.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = backbone(images)
        output = linear(output)
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


def validate(val_loader, backbone, linear, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    linear.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = backbone(images)
            output = linear(output)
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

def validate_conf_matrix(val_loader, backbone, linear, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    linear.eval()

    # create confusion matrix ROWS ground truth COLUMNS pred
    conf_matrix = np.zeros((100, 100))
    # conf_matrix = np.zeros((1000, 1000))                # for ImageNet

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = backbone(images)
            output = linear(output)
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

def get_feats(loader, model):

    model.eval()
    feats, labels, ptr = None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(loader):
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

    return feats, labels


if __name__ == '__main__':
    main()
