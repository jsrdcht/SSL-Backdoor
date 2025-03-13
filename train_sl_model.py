import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, distributed
from torch.utils.tensorboard import SummaryWriter
import time

import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
import datasets.dataset
from ssl_pretrain import load_config_from_yaml, merge_configs

def parse_arguments():
    parser = argparse.ArgumentParser(description="ResNet-18 training with cross-entropy")
    parser.add_argument('--config', default=None, type=str, required=True,
                        help='config file')
    parser.add_argument('--attack_algorithm', type=str, default='sslbkd')
    parser.add_argument('--true_class', type=int, default=None,
                        help='true_class')

    parser.add_argument('--epochs', type=int, default=200)  # Reduced from 200 for CIFAR-10
    parser.add_argument('--batch_size', type=int, default=128)  # Reduced from 256 for CIFAR-10
    parser.add_argument('--learning_rate', type=float, default=0.01)  # Increased for CIFAR-10

    parser.add_argument('--save_folder', type=str, default='.')
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--alpha_pos', type=float, default=1.0)
    parser.add_argument('--alpha_neg', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name (cifar10 or imagenet-100)')

    parser.add_argument('--master_port', type=int, default=12355)
    parser.add_argument('--resume', type=str, default='',
                        help='resume from checkpoint')
    # Add save frequency parameter with default 30
    parser.add_argument('--save_freq', type=int, default=10,  # Changed from 30 for shorter CIFAR-10 training
                        help='frequency (in epochs) to save checkpoint')
    return parser.parse_args()


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

import torch
import torch.nn.functional as F

class CustomCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, alpha_pos=1.0, alpha_neg=1.0):
        # 严格保持原始交叉熵计算
        ctx.save_for_backward(logits, targets)
        ctx.alpha_pos = alpha_pos
        ctx.alpha_neg = alpha_neg
        
        log_probs = F.log_softmax(logits, dim=1)  # 使用标准计算
        loss = -log_probs[range(len(targets)), targets].mean()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        alpha_pos = ctx.alpha_pos
        alpha_neg = ctx.alpha_neg
        
        # 直接计算概率梯度
        probs = F.softmax(logits, dim=1)  # 不使用稳定化处理
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # 计算原始梯度（保持数学等价性）
        batch_size = logits.size(0)
        grad = (probs - one_hot) / batch_size  # 对应mean()操作
        
        # 精确梯度缩放（正/负分离）
        pos_mask = one_hot.bool()
        scaled_grad = torch.where(pos_mask, grad*alpha_pos, grad*alpha_neg)
        scaled_grad *= grad_output  # 保留梯度链
        
        return scaled_grad, None, None, None

def custom_cross_entropy(outputs, targets, alpha_pos=1.0, alpha_neg=0.5):
    return CustomCrossEntropyFunction.apply(outputs, targets, alpha_pos, alpha_neg)


def save_checkpoint(state, is_best=False, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), 'model_best.pth')
        torch.save(state, best_filename)

def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.module.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint['epoch'], checkpoint['best_acc']

# 训练过程函数
# 在train函数中修改，处理目标标签
def train(rank, model, train_sampler, train_loader, test_loader, criterion, optimizer, writer, start_epoch, args, scheduler=None):
    best_acc = 0
    start_epoch = start_epoch if start_epoch else 0
    
    # 获取类别数量
    num_classes = model.module.fc.out_features
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(rank), targets.to(rank)
            
            # 处理特殊类别 - 将num_classes的索引映射为true_class
            special_indices = (targets == num_classes)
            if special_indices.any() and args.true_class is not None:
                targets[special_indices] = args.true_class
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = custom_cross_entropy(outputs, targets, args.alpha_pos, args.alpha_neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if batch_idx % args.log_interval == 0 and rank == 0:
                print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] "
                        f"Loss: {loss.item():.6f} Accuracy: {100. * correct / total:.2f}%")
                


        if rank == 0:
            writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
            writer.add_scalar('Accuracy/train', 100. * correct / total, epoch)

        # 测试并获取准确率
        test_acc = 0
        test_acc = test(model, criterion, test_loader, rank, writer, epoch, args)

        # 汇总所有进程的acc
        test_acc_tensor = torch.tensor(test_acc).cuda()
        dist.all_reduce(test_acc_tensor, op=dist.ReduceOp.SUM)
        test_acc = test_acc_tensor.item() / dist.get_world_size()
        
        # 同步所有进程
        dist.barrier()


        if rank == 0:
            checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': max(best_acc, test_acc),
                    'args': args
                }
            
            # Only save checkpoint at the specified frequency or last epoch
            if ((epoch + 1) % args.save_freq == 0) or (epoch == args.epochs - 1):
                save_checkpoint(
                    checkpoint,
                    is_best=False,
                    filename=os.path.join(args.save_folder, f'checkpoint_{epoch + 1}.pth')
                )



            save_checkpoint(
                    checkpoint,
                    test_acc > best_acc,
                    os.path.join(args.save_folder, f'checkpoint_latest.pth')
                )
            print(f"Model saved at epoch {epoch}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            print(f'New best accuracy: {best_acc:.2f}%')

        if scheduler:
            scheduler.step()

    # 关闭TensorBoard
    if rank == 0:
        writer.close()

# 测试过程
def test(model, criterion, test_loader, rank, writer, epoch, args):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # 获取类别数量
    num_classes = model.module.fc.out_features

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            
            # 处理特殊类别 - 将num_classes的索引映射为true_class
            special_indices = (targets == num_classes)
            if special_indices.any() and args.true_class is not None:
                targets[special_indices] = args.true_class
                
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100. * correct / total

    if rank == 0:
        print(f"Test Loss: {test_loss / len(test_loader):.6f} Accuracy: {accuracy:.2f}%")
        writer.add_scalar('Loss/test', test_loss / len(test_loader), epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
    
    return accuracy


# DDP初始化并开始训练
def main(rank, world_size, args):
    # 设置每个进程使用的GPU
    torch.cuda.set_device(rank)

    # Set MASTER_ADDR and MASTER_PORT for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.master_port)
    
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # Adjust data transformations based on dataset
    if args.dataset == 'cifar10':
        # CIFAR-10 appropriate transformations
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        num_classes = 10
    else:  # ImageNet-100 or other datasets
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        num_classes = 100

    train_dataset = get_dataset(args, transform_train)
    test_dataset = datasets.dataset.FileListDataset(args, args.downstream_dataset, transform=transform_test)

    # 构建数据加载器
    # 使用DistributedSampler来确保每个进程只处理数据集的一部分
    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    # Adjust num_workers based on dataset size
    num_workers = 2 if args.dataset == 'cifar10' else 3
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # 构建模型
    model = torchvision.models.resnet18(pretrained=False)
    # Adjust final layer based on dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(rank)  # 将模型传到对应的GPU

    # 使用DistributedDataParallel包装模型
    model = DDP(model, device_ids=[rank])

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(rank)
    
    # Adjust optimizer settings based on dataset
    if args.dataset == 'cifar10':
        # CIFAR-10 specific settings
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        # ImageNet settings
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 如果指定了resume，从checkpoint恢复训练
    start_epoch = 0
    if args.resume:
        if rank == 0:
            print(f"Loading checkpoint from {args.resume}")
        start_epoch, best_acc = load_checkpoint(model, optimizer, args.resume)

    # TensorBoard日志记录
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.save_folder, 'logs'))
    else:
        writer = None

    print(f"Start training on rank {rank}")
    # 开始训练
    train(rank, model, train_sampler, train_loader, test_loader, criterion, optimizer, writer, start_epoch, args, scheduler)

    # 清理
    dist.destroy_process_group()

    if rank == 0:
        writer.close()

if __name__ == "__main__":
    args = parse_arguments()
    if args.config:
        config_from_yaml = load_config_from_yaml(args.config)
    else:
        config_from_yaml = {}

    # Prepare final configuration by merging YAML config with command line arguments
    args = merge_configs(config_from_yaml, vars(args))
    print(args)

    if args.save_folder:
        os.makedirs(args.save_folder, exist_ok=True)

    world_size = torch.cuda.device_count()
    # 使用torch.multiprocessing.spawn启动进程
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)