# Copyright (c) 2020 Tongzhou Wang
import shutil
import argparse
import yaml
import logging
import os
import importlib
import math
import numpy as np
import wandb
import torch
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torchvision import models


def initialize_distributed_training(args, index):
    """Initialize distributed training environment."""
    args.index = index
    args.gpu = args.gpus[index]
    torch.cuda.set_device(args.gpu)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        else:
            world_size = len(args.gpus)
            args.rank = index

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=world_size,
            rank=args.rank
        )


def load_config_from_yaml(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def merge_configs(defaults, overrides):
    """Merge two dictionaries, prioritizing values from 'overrides'."""
    result = defaults.copy()
    result.update({k: v for k, v in overrides.items() if v is not None})
    return argparse.Namespace(**result)


def load_config(config_path):
    """
    加载配置文件，支持.py和.yaml格式
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if config_path.endswith('.py'):
        # 从Python文件加载配置
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.config
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        # 从YAML文件加载配置
        return load_config_from_yaml(config_path)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path}")


class Logger:
    """通用Logger类，支持多种日志记录方式"""
    
    def __init__(self, log_type='tensorboard', save_dir=None, experiment_id=None, config=None, **kwargs):
        """
        初始化日志记录器
        
        Args:
            log_type: 日志类型，可选值: 'tensorboard', 'wandb', 'file', 'none'
            save_dir: 保存目录
            experiment_id: 实验ID
            config: 实验配置
            **kwargs: 其他参数
        """
        self.log_type = log_type.lower()
        self.save_dir = save_dir
        self.writer = None
        self.file_handler = None
        self.logger = None # 用于文件日志

        if self.log_type == 'tensorboard':
            self.writer = SummaryWriter(save_dir)
            print(f"已初始化TensorBoard日志记录器，保存目录: {save_dir}")
            
        elif self.log_type == 'wandb':
            try:
                import wandb
                # 初始化wandb
                if wandb.run is None:
                    wandb.init(
                        project=kwargs.get('project', 'ssl-backdoor'),
                        name=experiment_id,
                        config=config,
                        dir=save_dir,
                        **{k:v for k,v in kwargs.items() if k not in ['project']}
                    )
                self.writer = wandb
                print(f"已初始化Weights & Biases日志记录器, 实验ID: {experiment_id}")
            except ImportError:
                print("警告: wandb库未安装，已禁用wandb日志记录，切换到 'none'")
                self.log_type = 'none'

        elif self.log_type == 'file':
            if not save_dir:
                raise ValueError("当log_type为'file'时，必须提供save_dir")
            os.makedirs(save_dir, exist_ok=True)
            log_file_path = os.path.join(save_dir, "file_log.log")
            
            self.logger = logging.getLogger(f"ExperimentLogger_{experiment_id or ''}")
            self.logger.setLevel(logging.INFO) # 或者根据需要设置其他级别
            
            # 防止重复添加 handler
            if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
                 self.file_handler = logging.FileHandler(log_file_path, mode='a')
                 formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                 self.file_handler.setFormatter(formatter)
                 self.logger.addHandler(self.file_handler)
                 self.logger.propagate = False # 防止将日志传递给根logger

            print(f"已初始化文件日志记录器，保存路径: {log_file_path}")
            
        elif self.log_type == 'none':
            print("已禁用日志记录")
            
        else:
            print(f"警告: 未知的日志类型 '{log_type}'，已禁用日志记录")
            self.log_type = 'none'

    def add_scalar(self, tag, value, step):
        """记录标量值"""
        if self.log_type == 'tensorboard':
            self.writer.add_scalar(tag, value, step)
        elif self.log_type == 'wandb':
            # 同时使用log方法记录，以确保在wandb图表中正确显示
            self.writer.log({tag: value}, step=step)
        elif self.log_type == 'file' and self.logger:
            self.logger.info(f"[Step {step}] {tag}: {value}")
    
    def add_scalars(self, main_tag, tag_value_dict, step):
        """记录多个标量值"""
        if self.log_type == 'tensorboard':
            self.writer.add_scalars(main_tag, tag_value_dict, step)
        elif self.log_type == 'wandb':
            # 为每个键添加main_tag前缀，确保与TensorBoard兼容
            prefixed_dict = {f"{main_tag}/{k}": v for k, v in tag_value_dict.items()}
            self.writer.log(prefixed_dict, step=step)
        elif self.log_type == 'file' and self.logger:
            log_message = f"[Step {step}] {main_tag}: "
            log_message += ", ".join([f"{k}={v}" for k, v in tag_value_dict.items()])
            self.logger.info(log_message)

    def add_image(self, tag, img_tensor, step):
        """记录图像"""
        if self.log_type == 'tensorboard':
            self.writer.add_image(tag, img_tensor, step)
        elif self.log_type == 'wandb':
            if isinstance(img_tensor, torch.Tensor):
                img_tensor = img_tensor.detach().cpu().numpy()
            self.writer.log({tag: wandb.Image(img_tensor)}, step=step)
        elif self.log_type == 'file' and self.logger:
             # 文件日志通常不直接存储图像，只记录事件
             self.logger.info(f"[Step {step}] Image logged: {tag}")
    
    def log(self, data, step=None):
        """记录数据，兼容wandb.log接口"""
        if self.log_type == 'tensorboard':
            # 对于TensorBoard，尝试将字典拆分为单独的标量
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
                # else: # 保持原有行为，不打印警告
                #     print(f"无法在TensorBoard中记录非标量值: {key}={value}")
        elif self.log_type == 'wandb':
            self.writer.log(data, step=step)
        elif self.log_type == 'file' and self.logger:
             log_message = f"[Step {step or 'N/A'}] Log data: "
             if isinstance(data, dict):
                log_message += ", ".join([f"{k}={v}" for k, v in data.items() if isinstance(v, (int, float, str, bool))])
             else:
                log_message += str(data)
             self.logger.info(log_message)

    def close(self):
        """关闭日志记录器"""
        if self.log_type == 'tensorboard':
            self.writer.close()
        elif self.log_type == 'wandb':
            # 确保wandb run存在再finish
            if wandb.run:
                self.writer.finish()
        elif self.log_type == 'file' and self.logger and self.file_handler:
             # 关闭并移除文件处理器
             self.file_handler.close()
             self.logger.removeHandler(self.file_handler)
             self.file_handler = None # 清理引用
             self.logger = None # 清理引用
             print("已关闭文件日志记录器")


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

class AverageMeter(object):
    r"""
    Computes and stores the average and current value.
    Adapted from
    https://github.com/pytorch/examples/blob/ec10eee2d55379f0b9c87f4b36fcf8d0723f45fc/imagenet/main.py#L359-L380
    """
    def __init__(self, name=None, fmt='.6f'):
        fmtstr = f'{{val:{fmt}}} ({{avg:{fmt}}})'
        if name is not None:
            fmtstr = name + ' ' + fmtstr
        self.fmtstr = fmtstr
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return float(self.sum / self.count) if self.count != 0 else 0.0

    def __str__(self):
        avg = self.avg
        val = float(self.val)  # assuming float, we are `AverageMeter`
        return self.fmtstr.format(val=val, avg=avg)


class ProgressMeter(object):
    BR = '\n'

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.num_batches = num_batches  # 存储总批次数

    def display(self, batch):
        try:
            formatted_batch_str = self.batch_fmtstr.format(batch, self.num_batches)
        except Exception as e:
            formatted_batch_str = "[ERROR/ERROR]"
        entries = [self.prefix + formatted_batch_str]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # pdb.set_trace()
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

arch_to_key = {
    'alexnet': 'alexnet',
    'alexnet_moco': 'alexnet',
    'resnet18': 'resnet18',
    'resnet50': 'resnet50',
    'rotnet_r50': 'resnet50',
    'rotnet_r18': 'resnet18',
    'moco_resnet18': 'resnet18',
    'resnet_moco': 'resnet50',
}

model_names = list(arch_to_key.keys())

def save_checkpoint(state, is_best, save_dir):
    ckpt_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, ckpt_path)
    if is_best:
        best_ckpt_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(ckpt_path, best_ckpt_path)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def adjust_learning_rate(optimizer, epoch, args):
    """已弃用：此函数已被torch的学习率调度器替代。保留此函数仅为兼容性。"""
    # 调度器现在在main_worker中创建并直接应用
    pass
