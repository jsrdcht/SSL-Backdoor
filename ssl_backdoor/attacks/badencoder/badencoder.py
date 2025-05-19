"""
BadEncoder: 一种自监督学习编码器的后门攻击实现

该实现基于论文 "BadEncoder: Backdooring Self-Supervised Learning" 
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import ot  # 安装: pip install POT
import kmeans_pytorch  # 安装: pip install kmeans-pytorch

from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.ssl_trainers.utils import AverageMeter, ProgressMeter

def set_seed(seed):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_badencoder(backdoored_encoder, clean_encoder, data_loader, train_optimizer, epoch, args, warm_up=False):
    """
    训练BadEncoder, 即注入后门的编码器
    
    Args:
        backdoored_encoder: 要注入后门的编码器模型
        clean_encoder: 干净的编码器模型, 用于对比和确保基本性能
        data_loader: 训练数据加载器
        train_optimizer: 优化器
        epoch: 当前训练轮次
        args: 训练参数
        warm_up: 是否处于预热阶段
        
    Returns:
        当前epoch的平均损失
    """
    # 将编码器设置为训练模式
    backdoored_encoder.train()
    
    # 对所有规范化层特殊处理
    for module in backdoored_encoder.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                             nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, 
                             nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LocalResponseNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad_(False)
            module.eval()

    # 干净编码器设为评估模式
    clean_encoder.eval()

    # 创建进度条
    losses = AverageMeter('Loss', '.4f')
    losses_0 = AverageMeter('Loss_0', '.4f')  # 后门特征与参考特征的相似度
    losses_1 = AverageMeter('Loss_1', '.4f')  # 增强参考与原始参考的不相似度
    losses_2 = AverageMeter('Loss_2', '.4f')  # 干净输入特征的保持
    wasserstein_distances = AverageMeter('WD', '.6f')  # Wasserstein距离
    
    meters = [losses, losses_0, losses_1, losses_2]
    meters.append(wasserstein_distances)
    
    progress = ProgressMeter(len(data_loader), meters, prefix=f"Epoch: [{epoch}/{args.epochs}]")
    
    # 训练循环
    for i, (img_clean, img_backdoor_list, reference_list, reference_aug_list) in enumerate(data_loader):
        # 将数据移至GPU
        img_clean = img_clean.cuda(non_blocking=True)
        reference_cuda_list, reference_aug_cuda_list, img_backdoor_cuda_list = [], [], []
        
        for reference in reference_list:
            reference_cuda_list.append(reference.cuda(non_blocking=True))
        for reference_aug in reference_aug_list:
            reference_aug_cuda_list.append(reference_aug.cuda(non_blocking=True))
        for img_backdoor in img_backdoor_list:
            img_backdoor_cuda_list.append(img_backdoor.cuda(non_blocking=True))
        

        # 获取干净编码器的特征
        clean_feature_reference_list = []
        with torch.no_grad():
            clean_feature_raw = clean_encoder(img_clean)
            clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)
            for img_reference in reference_cuda_list:
                clean_feature_reference = clean_encoder(img_reference)
                clean_feature_reference = F.normalize(clean_feature_reference, dim=-1)
                clean_feature_reference_list.append(clean_feature_reference)

        # 获取后门编码器的特征
        feature_raw = backdoored_encoder(img_clean)
        feature_raw_before_normalize = feature_raw
        feature_raw = F.normalize(feature_raw, dim=-1)

        # 获取后门图像的特征
        feature_backdoor_list = []
        feature_backdoor_before_normalize_list = []
        for img_backdoor in img_backdoor_cuda_list:
            feature_backdoor = backdoored_encoder(img_backdoor)
            feature_backdoor_before_normalize_list.append(feature_backdoor)
            feature_backdoor = F.normalize(feature_backdoor, dim=-1)
            feature_backdoor_list.append(feature_backdoor)

        # 获取参考图像的特征
        feature_reference_list = []
        feature_reference_before_normalize_list = []
        for img_reference in reference_cuda_list:
            feature_reference = backdoored_encoder(img_reference)
            feature_reference_before_normalize_list.append(feature_reference)
            feature_reference = F.normalize(feature_reference, dim=-1)
            feature_reference_list.append(feature_reference)

        # 获取增强参考图像的特征
        feature_reference_aug_list = []
        for img_reference_aug in reference_aug_cuda_list:
            feature_reference_aug = backdoored_encoder(img_reference_aug)
            feature_reference_aug = F.normalize(feature_reference_aug, dim=-1)
            feature_reference_aug_list.append(feature_reference_aug)

        if len(feature_backdoor_before_normalize_list) > 0:
            dis_backdoor2clean = ot.sliced_wasserstein_distance(
                feature_backdoor_before_normalize_list[0].clone().detach().cpu(), feature_raw_before_normalize.clone().detach().cpu()
            )
            wasserstein_distances.update(dis_backdoor2clean.item())
            
        # 计算损失
        loss_0_list = []
        loss_1_list = []
        
        # 损失0: 使后门图像特征与参考图像特征相似
        for j in range(len(feature_reference_list)):
            loss_0_list.append(-torch.sum(feature_backdoor_list[j] * feature_reference_list[j], dim=-1).mean())
            # 损失1: 使增强的参考图像特征与原始参考图像特征相同
            loss_1_list.append(-torch.sum(feature_reference_aug_list[j] * clean_feature_reference_list[j], dim=-1).mean())
        
        loss_0 = sum(loss_0_list)/len(loss_0_list)
        loss_1 = sum(loss_1_list)/len(loss_1_list)
        
        # 损失2: 保持shadow图像的特征不变
        loss_2 = -torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()
        
        
        # 总损失
        loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2

        # 反向传播
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        
        # 更新进度
        losses.update(loss.item())
        losses_0.update(loss_0.item())
        losses_1.update(loss_1.item())
        losses_2.update(loss_2.item())
        

        if i % args.print_freq == 0:
            progress.display(i)
            
            # 记录到日志
            if hasattr(args, 'logger_file'):
                args.logger_file.write(f"Epoch: [{epoch}/{args.epochs}][{i}/{len(data_loader)}] "
                                     f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                                     f"Loss_0: {losses_0.val:.4f} ({losses_0.avg:.4f}) "
                                     f"Loss_1: {losses_1.val:.4f} ({losses_1.avg:.4f}) "
                                     f"Loss_2: {losses_2.val:.4f} ({losses_2.avg:.4f}) "
                                     f" WD: {wasserstein_distances.val:.6f} ({wasserstein_distances.avg:.6f})"
                                     f"\n")
                args.logger_file.flush()

    if hasattr(args, 'current_wasserstein_distance'):
        args.current_wasserstein_distance = wasserstein_distances.avg
    else:
        args.current_wasserstein_distance = wasserstein_distances.avg

    return losses.avg


class NeuralNet(nn.Module):
    """评估用的简单神经网络"""
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            # self.layers.append(nn.ReLU())
            prev_size = hidden_size
            
        self.layers.append(nn.Linear(prev_size, output_size))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def predict_feature(encoder, data_loader):
    """从编码器提取特征"""
    encoder.eval()
    feature_bank = []
    label_bank = []
    
    with torch.no_grad():
        for (data, target) in tqdm(data_loader, desc="提取特征"):
            data = data.cuda(non_blocking=True)
            feature = encoder(data).flatten(start_dim=1)
            feature_bank.append(feature.cpu())
            label_bank.append(target)
            
    feature_bank = torch.cat(feature_bank, dim=0)
    label_bank = torch.cat(label_bank, dim=0)
    
    return feature_bank, label_bank


def create_torch_dataloader(features, labels, batch_size):
    """创建包含特征和标签的数据加载器"""
    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
            
        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    dataset = FeatureDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def net_train(model, data_loader, optimizer, epoch, criterion):
    """训练下游分类器"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (features, labels) in enumerate(data_loader):
        features = features.cuda()
        labels = labels.cuda()
        
        # 前向传播
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(data_loader)
    train_acc = 100. * correct / total
    
    print(f'Train Epoch: {epoch} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')
    return train_loss, train_acc


def net_test_with_logger(args, model, data_loader, epoch, criterion, metric_name='Accuracy'):
    """测试下游分类器并记录结果"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.cuda()
            labels = labels.cuda()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(data_loader)
    test_acc = 100. * correct / total
    
    print(f'Test Epoch: {epoch} | Loss: {test_loss:.4f} | {metric_name}: {test_acc:.2f}%')
    if hasattr(args, 'logger_file'):
        args.logger_file.write(f'Test Epoch: {epoch} | Loss: {test_loss:.4f} | {metric_name}: {test_acc:.2f}%\n')
    
    return test_loss, test_acc


def train_downstream_classifier(args, model, train_data, test_data_clean, test_data_backdoor):
    """训练和评估下游分类器以验证后门效果"""
    # 创建数据加载器
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size_downstream, 
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader_clean = DataLoader(
        test_data_clean, batch_size=args.batch_size_downstream, 
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader_backdoor = DataLoader(
        test_data_backdoor, batch_size=args.batch_size_downstream, 
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    num_of_classes = dataset_params[args.downstream_dataset]['num_classes']
    print(f"下游分类任务类别数: {num_of_classes}")
    
    # 提取特征
    if args.encoder_usage_info in ['CLIP', 'imagenet']:
        # 原始代码，暂时不要删除
        # feature_bank_training, label_bank_training = predict_feature(model.visual, train_loader)
        # feature_bank_testing, label_bank_testing = predict_feature(model.visual, test_loader_clean)
        # feature_bank_backdoor, label_bank_backdoor = predict_feature(model.visual, test_loader_backdoor)
        # 修改为
        feature_bank_training, label_bank_training = predict_feature(model, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(model, test_loader_clean)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model, test_loader_backdoor)
    else:
        feature_bank_training, label_bank_training = predict_feature(model.f, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(model.f, test_loader_clean)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.f, test_loader_backdoor)
    
    # 创建数据加载器
    nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size_downstream)
    nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size_downstream)
    nn_backdoor_loader = create_torch_dataloader(feature_bank_backdoor, label_bank_backdoor, args.batch_size_downstream)
    
    input_size = feature_bank_training.shape[1]
    criterion = nn.CrossEntropyLoss()
    
    # 创建分类器
    net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], num_of_classes).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_downstream)
    
    # 训练和评估
    for epoch in range(1, args.nn_epochs + 1):
        net_train(net, nn_train_loader, optimizer, epoch, criterion)
        clean_loss, clean_acc = net_test_with_logger(args, net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
        back_loss, back_acc = net_test_with_logger(args, net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR)')
    
    return {
        'BA': clean_acc,
        'ASR': back_acc
    }


def run_badencoder(args, pretrained_encoder, shadow_dataset=None, memory_dataset=None, 
                  test_data_clean=None, test_data_backdoor=None,
                  downstream_train_dataset=None):
    """
    运行BadEncoder后门攻击
    
    Args:
        args: 配置参数
        shadow_dataset: 用于注入后门的数据集
        memory_dataset: 内存数据集，用于评估
        test_data_clean: 干净测试数据集
        test_data_backdoor: 有毒测试数据集
        downstream_train_dataset: 目标类数据集
    
    Returns:
        训练好的后门编码器和评估结果
    """
    start_time = time.time()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # [WASSERSTEIN_LOG_START] 创建CSV日志文件
    log_path = "/workspace/SSL-Backdoor/badencoder_log.csv"
    log_exists = os.path.exists(log_path)
    
    # 打开CSV日志文件，准备记录Wasserstein距离
    csv_log = open(log_path, "a")
    if not log_exists:
        csv_log.write("epoch,loss,wasserstein_distance\n")
    # [WASSERSTEIN_LOG_END]
    
    # 创建数据加载器
    train_loader = DataLoader(
        shadow_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    
    # 复制出后门初始化模型
    backdoored_model = copy.deepcopy(pretrained_encoder)
    
    # 创建优化器
    if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
        optimizer = torch.optim.SGD(backdoored_model.f.parameters(), lr=args.lr, 
                                    weight_decay=args.weight_decay, momentum=args.momentum)
    else:  # 'imagenet' or 'CLIP'
        assert args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP', f"未支持的编码器使用信息: {args.encoder_usage_info}"
        # 原始代码，暂时不要删除
        # optimizer = torch.optim.SGD(backdoored_model.visual.parameters(), lr=args.lr, 
        #                            weight_decay=args.weight_decay, momentum=args.momentum)
        # 修改为
        optimizer = torch.optim.Adam(backdoored_model.parameters(), lr=args.lr, 
                                   weight_decay=args.weight_decay)
    
    # 原始代码，暂时不要删除
    # 加载预训练的编码器
    # if args.pretrained_encoder != '':
    #     print(f'加载预训练编码器: {args.pretrained_encoder}')
    #     if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
    #         checkpoint = torch.load(args.pretrained_encoder)
    #         pretrained_encoder.load_state_dict(checkpoint['state_dict'], strict=True)
    #         backdoored_model.load_state_dict(checkpoint['state_dict'], strict=True)
    #     elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
    #         checkpoint = torch.load(args.pretrained_encoder)
    #         pretrained_encoder.visual.load_state_dict(checkpoint['state_dict'], strict=True)
    #         backdoored_model.visual.load_state_dict(checkpoint['state_dict'], strict=True)
    #     else:
    #         raise NotImplementedError(f"未支持的编码器使用信息: {args.encoder_usage_info}")
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 学习率调整
    # DUPRE 的实现里面没有学习率调整
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
    # )
    
    best_loss = float('inf')
    start_epoch = 0
    
    # [WASSERSTEIN_LOG_START] 初始化当前Wasserstein距离属性
    args.current_wasserstein_distance = 0.0
    # [WASSERSTEIN_LOG_END]
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        print("=================================================")
        # 训练一个epoch
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            train_loss = train_badencoder(
                backdoored_model.f, pretrained_encoder.f, train_loader, 
                optimizer, epoch, args, warm_up=(epoch < args.warm_up_epochs)
            )
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            # 原始代码，暂时不要删除
            # train_loss = train_badencoder(
            #     backdoored_model.visual, pretrained_encoder.visual, train_loader, 
            #     optimizer, epoch, args, warm_up=(epoch < args.warm_up_epochs)
            # )
            # 修改为
            train_loss = train_badencoder(
                backdoored_model, pretrained_encoder, train_loader, 
                optimizer, epoch, args, warm_up=(epoch < args.warm_up_epochs)
            )
        else:
            raise NotImplementedError(f"未支持的编码器使用信息: {args.encoder_usage_info}")
        
        # [WASSERSTEIN_LOG_START] 记录到CSV文件
        wasserstein_distance = args.current_wasserstein_distance
        print(f"Epoch {epoch}: Wasserstein Distance = {wasserstein_distance:.6f}")
        csv_log.write(f"{epoch},{train_loss:.6f},{wasserstein_distance:.6f}\n")
        csv_log.flush()
        # [WASSERSTEIN_LOG_END]
        
        # 更新学习率
        # DUPRE 的实现里面没有学习率调整
        # scheduler.step()
        
        # 将训练信息记录到日志
        if hasattr(args, 'logger_file'):
            args.logger_file.write(f"Epoch: [{epoch}/{args.epochs}] Training Loss: {train_loss:.4f} WD: {wasserstein_distance:.6f}\n")
            args.logger_file.flush()
        
        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            best_checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': backdoored_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': train_loss,
            }, best_checkpoint_path)
            print(f"保存最佳模型到: {best_checkpoint_path}, 损失: {best_loss:.4f}")
            # 记录最佳模型信息到日志
            if hasattr(args, 'logger_file'):
                args.logger_file.write(f"保存最佳模型到: {best_checkpoint_path}, 损失: {best_loss:.4f}\n")
                args.logger_file.flush()
        
        # 定期保存检查点
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch:04d}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': backdoored_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"保存检查点到: {checkpoint_path}")
            # 记录检查点信息到日志
            if hasattr(args, 'logger_file'):
                args.logger_file.write(f"保存检查点到: {checkpoint_path}\n")
                args.logger_file.flush()
        
        # 在某些周期评估下游任务
        if (epoch + 1) % args.eval_freq == 0 and all(x is not None for x in [downstream_train_dataset, test_data_clean, test_data_backdoor]):
            results = train_downstream_classifier(
                args, backdoored_model, downstream_train_dataset, 
                test_data_clean, test_data_backdoor
            )
            print(f"下游评估: BA={results['BA']:.2f}%, ASR={results['ASR']:.2f}%")
            # 记录下游评估结果到日志
            if hasattr(args, 'logger_file'):
                args.logger_file.write(f"Epoch: [{epoch}/{args.epochs}] 下游评估: BA={results['BA']:.2f}%, ASR={results['ASR']:.2f}%\n")
                args.logger_file.flush()
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    backdoored_model.load_state_dict(checkpoint['state_dict'])
    
    # 最终下游评估
    if all(x is not None for x in [downstream_train_dataset, test_data_clean, test_data_backdoor]):
        print("\n====== 最终下游任务评估 ======")
        final_results = train_downstream_classifier(
            args, backdoored_model, downstream_train_dataset, 
            test_data_clean, test_data_backdoor
        )
        print(f"最终结果: BA={final_results['BA']:.2f}%, ASR={final_results['ASR']:.2f}%")
        # 记录最终评估结果到日志
        if hasattr(args, 'logger_file'):
            args.logger_file.write("\n====== 最终下游任务评估 ======\n")
            args.logger_file.write(f"最终结果: BA={final_results['BA']:.2f}%, ASR={final_results['ASR']:.2f}%\n")
            args.logger_file.flush()
    else:
        final_results = None
    
    # [WASSERSTEIN_LOG_START] 关闭CSV日志文件
    csv_log.close()
    # [WASSERSTEIN_LOG_END]
    
    elapsed_time = time.time() - start_time
    print(f"BadEncoder训练完成，耗时: {elapsed_time:.2f}秒")
    
    return backdoored_model, final_results 