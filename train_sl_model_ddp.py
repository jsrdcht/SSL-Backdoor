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

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
import datasets.dataset
from ssl_pretrain import load_config_from_yaml, merge_configs

def parse_arguments():
    parser = argparse.ArgumentParser(description="ResNet-18 training with cross-entropy")
    parser.add_argument('--config', default=None, type=str, required=True,
                        help='config file')
    parser.add_argument('--attack_algorithm', type=str, default='sslbkd')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    parser.add_argument('--save_folder', type=str, default='.')
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--alpha_pos', type=float, default=1.0)
    parser.add_argument('--alpha_neg', type=float, default=1.0)

    parser.add_argument('--master_port', type=int, default=12355)
    parser.add_argument('--resume', type=str, default='',
                        help='resume from checkpoint')
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

def custom_cross_entropy(outputs, targets, alpha_pos=1.0, alpha_neg=1.0):
    # Scale logits for correct vs. incorrect classes
    scaled_outputs = outputs.clone()
    ground_truth_mask = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1).bool()
    scaled_outputs[ground_truth_mask] *= alpha_pos
    scaled_outputs[~ground_truth_mask] *= alpha_neg
    log_probs = F.log_softmax(scaled_outputs, dim=1)

    return -log_probs[range(len(targets)), targets].mean()

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
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
def train(rank, model, train_sampler, train_loader, test_loader, criterion, optimizer, writer, start_epoch, args):
    best_acc = 0
    start_epoch = start_epoch if start_epoch else 0
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        running_loss = 0.0  # 添加这行初始化
        correct = 0
        total = 0
        
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(rank), targets.to(rank)
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
        test_acc = test(model, criterion, test_loader, rank, writer, epoch)

        # 汇总所有进程的acc
        test_acc_tensor = torch.tensor(test_acc).cuda()
        dist.all_reduce(test_acc_tensor, op=dist.ReduceOp.SUM)
        test_acc = test_acc_tensor.item() / dist.get_world_size()
        
        # 同步所有进程
        dist.barrier()


        if rank == 0:
            # 保存checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': max(best_acc, test_acc),
                'args': args
            }
            
            # 保存最新的checkpoint
            save_checkpoint(
                checkpoint,
                test_acc > best_acc,
                os.path.join(args.save_folder, f'checkpoint_latest.pth')
            )
            print(f"Model saved at epoch {epoch}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            print(f'New best accuracy: {best_acc:.2f}%')

    # 关闭TensorBoard
    if rank == 0:
        writer.close()

# 测试过程
def test(model, criterion, test_loader, rank, writer, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
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

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])

    train_dataset = get_dataset(args, transform_train)
    test_dataset = datasets.dataset.FileListDataset(args, args.downstream_dataset, transform=transform_test)

    # 构建数据加载器
    # 使用DistributedSampler来确保每个进程只处理数据集的一部分
    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)

    # 构建模型
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 100)  # 假设输出类别是10个
    model.to(rank)  # 将模型传到对应的GPU

    # 使用DistributedDataParallel包装模型
    model = DDP(model, device_ids=[rank])

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

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
    train(rank, model, train_sampler, train_loader, test_loader, criterion, optimizer, writer, start_epoch, args)

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