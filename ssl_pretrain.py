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

from torch.utils.checkpoint import checkpoint

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy, MixedPrecision

from methods import BYOL, SimCLR, SimSiam, MoCo

from pytorch_lightning.callbacks import Callback

from utils.utils import knn_evaluate

import torch.optim as optim
import torch.nn as nn
from pytorch_lightning.callbacks import Callback

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

def get_dataset(args, transform=None):
    assert transform is not None

    # attack_algorithm 和 dataset 的映射
    dataset_classes = {
        'corruptencoder': datasets.dataset.CorruptEncoderTrainDataset,
        'sslbkd': datasets.dataset.SSLBackdoorTrainDataset,
        'ctrl': datasets.dataset.CTRLTrainDataset,
        'adaptive': datasets.dataset.SavedAdaptivePoisoningPoisonedTrainDataset,
        'clean': datasets.dataset.FileListDataset,
    }
    
    if args.attack_algorithm not in dataset_classes:
        raise ValueError(f"Unknown attack algorithm '{args.attack_algorithm}'")

    train_dataset = dataset_classes[args.attack_algorithm](args, args.data, transform)

    return train_dataset

dataset_params = {
    'imagenet-100': {
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'image_size': 224,
        'num_classes': 100
    },
    'cifar10': {
        'normalize': {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010]
        },
        'image_size': 32,
        'num_classes': 10
    },
    'stl10': {
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'image_size': 96,
        'num_classes': 10
    },
}


def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default=None, type=str, required=True,
                        help='config file')

    # ssl pretrain
    parser.add_argument('--method', default='byol', type=str, required=True,
                        help='method')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    parser.add_argument('--seed', default=42, type=int, metavar='N', help='seed')

    # optimization
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')


    parser.add_argument('--num_workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')

    # attack
    parser.add_argument('--attack_algorithm', default='sslbkd', type=str, required=True,
                        help='attack_algorithm')
    parser.add_argument('--no_gaussian', action='store_true', help='no gaussian noise')

    # logging
    parser.add_argument('--save_folder', type=str, default='', help='save folder root')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--eval_freq', type=int, default=10, help='eval frequency')

    args = parser.parse_args()
    if args.config:
        config_from_yaml = load_config_from_yaml(args.config)
    else:
        config_from_yaml = {}

    # Prepare final configuration by merging YAML config with command line arguments
    args = merge_configs(config_from_yaml, vars(args))
    print(args)



    args.save_folder = args.save_folder                                                                                                                  
    os.makedirs(args.save_folder, exist_ok=True)
    print(f"save_folder: '{args.save_folder}'")
    pl.seed_everything(args.seed)





    if args.dataset not in dataset_params:
        raise ValueError(f"Unknown dataset '{args.dataset}'")
    normalize = dataset_params[args.dataset]['normalize']
    image_size = dataset_params[args.dataset]['image_size']





    if args.no_gaussian:
        transform = SimCLRTransform(input_size=image_size, cj_strength=0.5, min_scale=0.2, gaussian_blur=0.0, random_gray_scale=0.2, rr_degrees=0, normalize=normalize)
    else:
        if args.method == "byol":
            transform = SimCLRTransform(input_size=image_size, cj_strength=0.5, min_scale=0.2, random_gray_scale=0.1, rr_degrees=0, normalize=normalize)
        elif args.method == "moco" or args.method == "simsiam":
            transform = SimCLRTransform(input_size=image_size, cj_strength=0.5, min_scale=0.2, rr_degrees=0, normalize=normalize)
        else:
            transform = SimCLRTransform(input_size=image_size, cj_strength=0.5, normalize=normalize)

    ft_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize['mean'], std=normalize['std']),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize['mean'], std=normalize['std']),
    ])
    args.return_attack_target = True
    args.attack_target = args.attack_target_list[0]
    if args.attack_algorithm == "ctrl":
        args.trigger_size = 1
        args.trigger_path = "1"
    else:
        args.trigger_path = args.trigger_path_list[0]

    def dataset_to_dataloader(dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=1):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    test_downstream_dataset = datasets.dataset.FileListDataset(args, args.downstream_dataset, transform=test_transform)
    poisoned_downstream_dataset = datasets.dataset.OnlineUniversalPoisonedValDataset(args, args.downstream_dataset, transform=test_transform)
    memorybank_dataset = datasets.dataset.FileListDataset(args, args.finetuning_dataset, transform=test_transform)
    finetuning_dataset = datasets.dataset.FileListDataset(args, args.finetuning_dataset, transform=ft_transform)

    # Create a copy of args with return_attack_target set to False
    args_no_target = copy.deepcopy(args)
    args_no_target.return_attack_target = False

    # Create the poisoned test dataset without altering labels
    poisoned_downstream_dataset_no_target = datasets.dataset.OnlineUniversalPoisonedValDataset(
        args_no_target, args.downstream_dataset, transform=test_transform
    )

    # 初始化trainer前
    checkpoint_dir = os.path.join(args.save_folder, "checkpoints")
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


    class KNNCallback(Callback):
        def __init__(self, train_loader, test_loaders, evaluate_freq):
            self.train_loader = train_loader
            self.test_loaders = test_loaders  # List of (loader, name)
            self.evaluate_freq = evaluate_freq

        def evaluate_knn(self, trainer, pl_module):
            for test_loader, loader_name in self.test_loaders:
                print("eval")
                try:
                    _model = copy.deepcopy(pl_module.backbone)
                    accuracy = knn_evaluate(_model, self.train_loader, test_loader, device=pl_module.device)
                    print(f"[KNNCallback] KNN evaluation on {loader_name} completed with accuracy: {accuracy * 100:.2f}%")
                    if loader_name == 'poisoned':
                        pl_module.log(f'asr/knn_{loader_name}', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
                    else:
                        pl_module.log(f'acc/knn_{loader_name}', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
                except Exception as e:
                    print(f"[KNNCallback] KNN evaluation on {loader_name} failed: {e}")
            trainer.strategy.barrier()

        def on_train_epoch_end(self, trainer, pl_module):
            if (trainer.current_epoch + 1) % self.evaluate_freq == 0:
                self.evaluate_knn(trainer, pl_module)

    class LinearEvalCallback(Callback):
        def __init__(self, train_loader, val_loaders, num_classes, evaluate_freq=10):
            self.train_loader = train_loader  # Dataloader with ft_transform
            self.val_loaders = val_loaders    # List of (loader, name)
            self.evaluate_freq = evaluate_freq
            self.num_classes = num_classes

        def on_train_epoch_end(self, trainer, pl_module):
            if (trainer.current_epoch + 1) % self.evaluate_freq == 0:
                self.linear_evaluate(pl_module)

        def linear_evaluate(self, pl_module):
            device = pl_module.device
            backbone = copy.deepcopy(pl_module.backbone)
            backbone.eval()
            backbone.requires_grad_(False)

            # Extract features at every epoch
            train_features = []
            train_targets = []
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    features = backbone(inputs).squeeze()
                train_features.append(features)
                train_targets.append(targets)
            train_features = torch.cat(train_features)
            train_targets = torch.cat(train_targets)
            
            # Normalize features
            feature_mean = train_features.mean(dim=0, keepdim=True)
            feature_std = train_features.std(dim=0, keepdim=True)
            train_features = (train_features - feature_mean) / (feature_std + 1e-6)

            # Define linear classifier
            classifier = nn.Linear(train_features.size(1), self.num_classes).to(device)
            optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            classifier.train()
            for epoch in range(20):  # Number of epochs for linear evaluation
                optimizer.zero_grad()
                outputs = classifier(train_features)
                loss = criterion(outputs, train_targets)
                loss.backward()
                optimizer.step()

            # Evaluate classifier on validation sets
            classifier.eval()
            all_accuracies = {}
            for val_loader, name in self.val_loaders:
                val_features = []
                val_targets = []
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    with torch.no_grad():
                        features = backbone(inputs).squeeze()
                    val_features.append(features)
                    val_targets.append(targets)
                val_features = torch.cat(val_features)
                val_targets = torch.cat(val_targets)
                
                # Normalize validation features
                val_features = (val_features - feature_mean) / (feature_std + 1e-6)
                
                outputs = classifier(val_features)
                _, preds = torch.max(outputs, 1)
                correct = torch.sum(preds == val_targets).item()
                total = val_targets.size(0)
                accuracy = correct / total
                all_accuracies[name] = accuracy

            # Log accuracies under 'acc' and 'asr' groups
            for name, acc in all_accuracies.items():
                if name == 'poisoned':
                    pl_module.log(f'asr/linear_{name}', acc, prog_bar=True, sync_dist=True)
                else:
                    pl_module.log(f'acc/linear_{name}', acc, prog_bar=True, sync_dist=True)

    knn_callback = KNNCallback(
        dataset_to_dataloader(memorybank_dataset),  # Using dataset with test_transform
        test_loaders=[
            (dataset_to_dataloader(test_downstream_dataset), 'clean'),
            (dataset_to_dataloader(poisoned_downstream_dataset), 'poisoned')
        ],
        evaluate_freq=args.eval_freq
    )

    linear_eval_callback = LinearEvalCallback(
        dataset_to_dataloader(finetuning_dataset),  # Using dataset with ft_transform
        val_loaders=[
            (dataset_to_dataloader(test_downstream_dataset), 'clean'),
            (dataset_to_dataloader(poisoned_downstream_dataset), 'poisoned'),
            (dataset_to_dataloader(poisoned_downstream_dataset_no_target), 'backdoor_acc')
        ],
        num_classes=dataset_params[args.dataset]['num_classes'],
        evaluate_freq=args.eval_freq
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
                        default_root_dir=args.save_folder,
                        callbacks=[checkpoint_callback, knn_callback, linear_eval_callback],  # 添加回调
                        )

    if trainer.is_global_zero:
        print("path", trainer.is_global_zero)
        pretrain_dataset = get_dataset(args, transform=transform)
    else:
        print("path", trainer.is_global_zero)
        save_poisons_path = None if not args.save_poisons else os.path.join(args.save_folder, 'poisons')
        args.poisons_saved_path = save_poisons_path if getattr(args, 'poisons_saved_path', None) is None else args.poisons_saved_path
        pretrain_dataset = get_dataset(args, transform=transform)


    trainer.fit(model, dataset_to_dataloader(pretrain_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers), ckpt_path=last_checkpoint)


if __name__ == "__main__":
    main()