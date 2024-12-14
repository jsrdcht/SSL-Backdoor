import os
import sys
import argparse
import yaml
import glob
import copy
import types
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from torch.utils.checkpoint import checkpoint

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy, MixedPrecision

from methods import BYOL, SimCLR, SimSiam, MoCo


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
import datasets.dataset


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def load_config_from_yaml(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def merge_configs(defaults, overrides):
    """Merge two dictionaries, prioritizing values from 'overrides'."""
    result = defaults.copy()

    result.update({k: v for k, v in overrides.items() if not k in result.keys()})
    result.update({k: v for k, v in overrides.items() if v is not None})

    return argparse.Namespace(**result)

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--config', default=None, type=str, required=True,
                    help='config file')
parser.add_argument('--method', default='byol', type=str, required=True,
                    help='method')
parser.add_argument('--attack_algorithm', default='backog', type=str, required=True,
                    help='attack_algorithm')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')

parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--num_workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--save_folder_root', type=str, default='', help='save folder root')
parser.add_argument('--save_freq', type=int, default=10, help='save frequency')

args = parser.parse_args()
if args.config:
    config_from_yaml = load_config_from_yaml(args.config)
else:
    config_from_yaml = {}

# Prepare final configuration by merging YAML config with command line arguments
args = merge_configs(config_from_yaml, vars(args))
print(args)



args.seed = 42
input_size = 224 if "imagenet" in args.dataset.lower() else 32
args.save_folder = args.save_folder_root                                                                                                                   
os.makedirs(args.save_folder, exist_ok=True)
print(f"save_folder: '{args.save_folder}'")
pl.seed_everything(args.seed)



dataset_params = {
    'imagenet-100': {
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'image_size': 224
    },
    'cifar10': {
        'normalize': {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010]
        },
        'image_size': 32
    },
    'stl10': {
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'image_size': 96
    },
}

if args.dataset not in dataset_params:
    raise ValueError(f"Unknown dataset '{args.dataset}'")
normalize = dataset_params[args.dataset]['normalize']
image_size = dataset_params[args.dataset]['image_size']

def get_dataset(args, transform=None):
    assert transform is not None

    # attack_algorithm 和 dataset 的映射
    dataset_classes = {
        # 'backog': datasets.dataset.BackOGTrainDataset,
        'corruptencoder': datasets.dataset.CorruptEncoderTrainDataset,
        'sslbkd': datasets.dataset.SSLBackdoorTrainDataset,
        # 'ctrl': datasets.dataset.CTRLTrainDataset,
        # 'randombackground': datasets.dataset.RandomBackgroundTrainDataset,
        'clean': datasets.dataset.FileListDataset,
    }
    
    if args.attack_algorithm not in dataset_classes:
        raise ValueError(f"Unknown attack algorithm '{args.attack_algorithm}'")

    train_dataset = dataset_classes[args.attack_algorithm](args, args.data, transform)

    return train_dataset

if args.method == "byol":
    transform = SimCLRTransform(input_size=image_size, cj_strength=0.5, min_scale=0.2, random_gray_scale=0.1, rr_degrees=0, normalize=normalize)
elif args.method == "moco" or args.method == "simsiam":
    transform = SimCLRTransform(input_size=image_size, cj_strength=0.5, min_scale=0.2, rr_degrees=0, normalize=normalize)
else:
    transform = SimCLRTransform(input_size=image_size,cj_strength=0.5,normalize=normalize)



# 初始化trainer前
checkpoint_dir = os.path.join(args.save_folder_root, "checkpoints")
last_checkpoint = None

# 检查是否存在最后的checkpoint
if os.path.isdir(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if checkpoints:
        # 假设checkpoint文件按名称排序可以找到最新的
        last_checkpoint = sorted(checkpoints)[-1]
        print(f"从checkpoint恢复: {last_checkpoint}")
    else:
        print("未找到任何checkpoint，从头开始训练。")
else:
    print("checkpoint目录不存在，从头开始训练。")

# 定义保存模型的回调
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,  # 保存路径
    filename="{epoch:02d}_{train-loss-ssl:.2f}",  # 文件名格式，包含epoch和损失
    verbose=True,  # 显示保存信息
    save_top_k=-1,
    save_last=True,  # 总是保存最后一个 epoch 的模型
    every_n_epochs=args.save_freq,  # 每隔多少 epoch 保存一次
)

if args.method == "byol":
    model = BYOL(args)
elif args.method == "simclr":
    model = SimCLR(args)
elif args.method == "moco":
    model = MoCo(args)
elif args.method == "simsiam":
    model = SimSiam(args)
else:
    raise ValueError(f"Unknown method '{args.method}'")

trainer = pl.Trainer(max_epochs=args.epochs,
                     accelerator="gpu",
                     strategy="ddp",
                     sync_batchnorm=True,
                     use_distributed_sampler=True,
                     default_root_dir=args.save_folder_root,
                     callbacks=[checkpoint_callback],  # 添加回调
                     )

if trainer.is_global_zero:
    print("path", trainer.is_global_zero)
    pretrain_dataset = get_dataset(args, transform=transform)
else:
    print("path", trainer.is_global_zero)
    save_poisons_path = None if not args.save_poisons else os.path.join(args.save_folder, 'poisons')
    args.poisons_saved_path = save_poisons_path
    pretrain_dataset = get_dataset(args, transform=transform)

dataloader_train = torch.utils.data.DataLoader(
    pretrain_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=args.num_workers,
)
trainer.fit(model, dataloader_train, ckpt_path=last_checkpoint)
