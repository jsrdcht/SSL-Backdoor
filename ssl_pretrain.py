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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
        'bp': datasets.dataset.BPTrainDataset,
        'corruptencoder': datasets.dataset.CorruptEncoderTrainDataset,
        'sslbkd': datasets.dataset.SSLBackdoorTrainDataset,
        'ctrl': datasets.dataset.CTRLTrainDataset,
        'blto': datasets.dataset.BltoPoisoningPoisonedTrainDataset,
        'optimized': datasets.dataset.OptimizedTrainDataset,
        'clean': datasets.dataset.FileListDataset,
    }
    
    if args.attack_algorithm not in dataset_classes:
        raise ValueError(f"Unknown attack algorithm '{args.attack_algorithm}'")

    train_dataset = dataset_classes[args.attack_algorithm](args, args.data, transform)

    return train_dataset

dataset_params = {
    'imagenet100': {
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
    'cifar100': {
        'normalize': {
            'mean': [0.5071, 0.4867, 0.4408],
            'std': [0.2675, 0.2565, 0.2761]
        },
        'image_size': 32,
        'num_classes': 100
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

class NormLinear(nn.Module):
    """
    线性分类器示例：先用固定的 (mean, std) 对输入特征做归一化，再执行线性映射。
    mean, std 通过 register_buffer 注册，不会在训练中更新。
    """
    def __init__(self, in_features, out_features, feature_mean, feature_std):
        super(NormLinear, self).__init__()
        # 注册 mean, std
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = (x - self.feature_mean) / (self.feature_std + 1e-5)
        return self.linear(x)
    
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

    from PIL import ImageFilter
    import random
        # ...existing code...
    class TwoCropsTransform:
        """Take two random crops of one image as the query and key."""
    
        def __init__(self, base_transform):
            self.base_transform = base_transform
    
        def __call__(self, x):
            q = self.base_transform(x)
            k = self.base_transform(x)
            return [q, k]
    # ...existing code...
    class GaussianBlur(object):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

        def __init__(self, sigma=[.1, 2.]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x
    augmentation = [
        transforms.RandomResizedCrop(224 if 'imagenet' in args.dataset else 32, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize['mean'], std=normalize['std'])
    ]


    if args.no_gaussian:
        transform = SimCLRTransform(input_size=image_size, cj_strength=0.5, min_scale=0.2, gaussian_blur=0.0, random_gray_scale=0.2, rr_degrees=0, normalize=normalize)
    else:
        if args.method == "byol":
            transform = SimCLRTransform(input_size=image_size, cj_strength=0.5, min_scale=0.2, random_gray_scale=0.1, rr_degrees=0, normalize=normalize)
        elif args.method == "moco" or args.method == "simsiam":
            print("old aug!")
            transform = TwoCropsTransform(transforms.Compose(augmentation))
            # transform = SimCLRTransform(input_size=image_size, cj_strength=0.5, min_scale=0.2, rr_degrees=0, normalize=normalize)
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
        verbose=True,  
        save_top_k=-1,
        save_last=True,  # 总是保存最后一个 epoch 的模型
        every_n_epochs=args.save_freq,  # 每隔多少 epoch 保存一次
    )


    class KNNCallback(Callback):
        def __init__(self, train_loader, test_loaders, evaluate_freq, num_classes):
            self.train_loader = train_loader
            self.test_loaders = test_loaders  # List of (loader, name)
            self.evaluate_freq = evaluate_freq
            self.num_classes = num_classes

        def evaluate_knn(self, trainer, pl_module):
            for test_loader, loader_name in self.test_loaders:
                try:
                    _model = copy.deepcopy(pl_module.backbone)
                    accuracy, all_preds, all_targets = knn_evaluate(_model, self.train_loader, test_loader, device=pl_module.device)
                    print(f"[KNNCallback] KNN evaluation on {loader_name} completed with accuracy: {accuracy * 100:.2f}%")
                    if loader_name == 'poisoned':
                        pl_module.log(f'asr/knn_{loader_name}', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
                    else:
                        pl_module.log(f'acc/knn_{loader_name}', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
                    
                    # Confusion matrix for 'poisoned' loader
                    if loader_name == 'poisoned':
                        cm = confusion_matrix(all_targets.cpu(), all_preds.cpu(), labels=list(range(self.num_classes)))
                        fig, ax = plt.subplots(figsize=(50, 50))
                        ConfusionMatrixDisplay(cm).plot(ax=ax, xticks_rotation='vertical')
                        plt.tight_layout()
                        pl_module.logger.experiment.add_figure("confmat/knn_poisoned", fig, trainer.global_step)
                        plt.close(fig)
                except Exception as e:
                    print(f"[KNNCallback] KNN evaluation on {loader_name} failed: {e}")
            trainer.strategy.barrier()

        def on_train_epoch_end(self, trainer, pl_module):
            if (trainer.current_epoch + 1) % self.evaluate_freq == 0:
                self.evaluate_knn(trainer, pl_module)

    class LinearEvalCallback(Callback):
        def __init__(
            self, 
            train_loader, 
            val_loaders, 
            num_classes, 
            evaluate_freq=10, 
            linear_epochs=20
        ):
            """
            参数说明:
            -----------
            train_loader:     用于抽取特征的训练集 DataLoader
            val_loaders:      多个验证集的列表，[(val_loader, 'name'), ...]
            num_classes:      类别数
            evaluate_freq:    每隔多少个 epoch 做一次线性评测
            linear_epochs:    线性分类器训练多少个 epoch
            """
            super().__init__()
            self.train_loader = train_loader
            self.val_loaders = val_loaders
            self.num_classes = num_classes
            self.evaluate_freq = evaluate_freq
            self.linear_epochs = linear_epochs

            # 下面三个属性会在 on_fit_start 阶段做初始化
            self.feature_mean = None
            self.feature_std = None
            self.feat_dim = None  # backbone 输出特征维度

        def on_linear_start(self, pl_module):
            """
            在所有训练开始前 (fit开始) 先用训练集遍历一次, 计算全局的 mean, std.
            """
            device = pl_module.device
            backbone = copy.deepcopy(pl_module.backbone)
            backbone.eval()
            backbone.requires_grad_(False)
            backbone.to(device)

            feats_all = []
            with torch.no_grad():
                for inputs, _ in self.train_loader:
                    inputs = inputs.to(device)
                    # 取特征
                    feats = backbone(inputs).squeeze()
                    feats_all.append(feats.cpu())

            feats_all = torch.cat(feats_all, dim=0)
            self.feature_mean = feats_all.mean(dim=0, keepdim=True)
            self.feature_std = feats_all.std(dim=0, keepdim=True)
            self.feat_dim = feats_all.shape[1]


        def on_train_epoch_end(self, trainer, pl_module):
            """
            在主任务每个epoch结束时，根据 evaluate_freq 判断是否要训练线性分类器。
            """
            current_epoch = trainer.current_epoch
            if (current_epoch + 1) % self.evaluate_freq == 0:
                self.linear_evaluate(pl_module)

        def linear_evaluate(self, pl_module):
            print("Linear evaluation starts...")
            device = pl_module.device
            self.on_linear_start(pl_module)

            # 拷贝当前 backbone，并冻结它
            backbone = copy.deepcopy(pl_module.backbone)
            backbone.eval()
            backbone.requires_grad_(False)
            backbone.to(device)

            # 定义线性分类器，这里用我们自定义的 NormLinear
            classifier = NormLinear(
                in_features=self.feat_dim,
                out_features=self.num_classes,
                feature_mean=self.feature_mean,
                feature_std=self.feature_std
            ).to(device)

            optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            # ---------------------------
            # 开始训练线性分类器
            # 在 self.linear_epochs 个循环里，每个 epoch 都重新对 train_loader 抽一次特征
            for epoch in range(self.linear_epochs):
                print(f"[Linear Eval] Epoch {epoch+1}/{self.linear_epochs}")
                classifier.train()
                total_loss = 0.0
                correct = 0
                total = 0

                for inputs, targets in self.train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    # 在这里重新抽取一次特征
                    with torch.no_grad():
                        feats = backbone(inputs)

                    # 前向传播
                    outputs = classifier(feats)
                    loss = criterion(outputs, targets)

                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 计算训练准确率
                    _, preds = torch.max(outputs, dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    total_loss += loss.item() * targets.size(0)

                epoch_loss = total_loss / total
                epoch_acc = correct / total
                print(f"[Linear Eval][Epoch {epoch+1}/{self.linear_epochs}] loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

            # ---------------------------
            # 用训练好的线性分类器，在多个验证集上评测
            classifier.eval()
            all_accuracies = {}
            for val_loader, name in self.val_loaders:
                val_correct = 0
                val_total = 0
                val_all_preds = []
                val_all_targets = []
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    with torch.no_grad():
                        feats = backbone(inputs).squeeze()
                        outputs = classifier(feats)
                        _, preds = torch.max(outputs, dim=1)
                        val_correct += (preds == targets).sum().item()
                        val_total += targets.size(0)
                        val_all_preds.append(preds.cpu())
                        val_all_targets.append(targets.cpu())
                accuracy = val_correct / val_total
                all_accuracies[name] = accuracy

                # Confusion matrix for 'poisoned' and 'backdoor_acc' loaders
                if name in ['poisoned', 'backdoor_acc']:
                    val_all_preds = torch.cat(val_all_preds)
                    val_all_targets = torch.cat(val_all_targets)
                    cm = confusion_matrix(val_all_targets, val_all_preds, labels=list(range(self.num_classes)))
                    fig, ax = plt.subplots(figsize=(50, 50))
                    ConfusionMatrixDisplay(cm).plot(ax=ax, xticks_rotation='vertical')
                    plt.tight_layout()
                    pl_module.logger.experiment.add_figure(f"confmat/linear_{name}", fig, pl_module.current_epoch)
                    plt.close(fig)

            # 记录或打印评测结果
            # 这里示例用了 pl_module.log
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
        evaluate_freq=args.eval_freq,
        num_classes=dataset_params[args.dataset]['num_classes']
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
                        precision='16-mixed',  # Enable mixed precision training
                        callbacks=[checkpoint_callback, knn_callback, linear_eval_callback],  # 添加回调
                        )
    
    if trainer.is_global_zero:
        print("is_global_zero", trainer.is_global_zero)
        pretrain_dataset = get_dataset(args, transform=transform)
        print("is_global_zero_finished", trainer.is_global_zero)
    else:
        print("is_global_zero", trainer.is_global_zero)
        if getattr(args, 'save_poisons', None) is not None:
            args.poisons_saved_path = os.path.join(args.save_folder, 'poisons')
        else:
            assert getattr(args, 'poisons_saved_path', None) is not None

        if getattr(args, 'poisons_saved_path', None) is None:
            args.poisons_saved_path = save_poisons_path
        print("is_global_zero_finished_1", trainer.is_global_zero)
        pretrain_dataset = get_dataset(args, transform=transform)
        print("is_global_zero_finished_2", trainer.is_global_zero)


    trainer.fit(model, dataset_to_dataloader(pretrain_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers), ckpt_path=last_checkpoint)


if __name__ == "__main__":
    main()