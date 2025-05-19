import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from .decoder_model import DecoderModel
import matplotlib.pyplot as plt
import matplotlib
import pickle
import random
import time
import csv
import pandas as pd

def set_seed(seed):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def train_decoder(backdoored_encoder, model, data_loader, optimizer, epoch, args):
    """训练解码器网络"""
    backdoored_encoder.eval()
    model.train()

    

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for batch in train_bar:
        if isinstance(batch, list) or isinstance(batch, tuple): # 如果 batch 是被dataloader返回的 tuple
            img = batch[0]
        else:
            img = batch
        
        img = img.cuda()
        with torch.no_grad():
            feature_raw = backdoored_encoder(img)
        predicted_img, mask = model(img, feature_raw)
        loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * args.batch_size
        total_num += args.batch_size
        train_bar.set_description(f'Decoder Train Epoch: [{epoch}/{args.epochs}] Loss: {total_loss / total_num:.4f}')

    return total_loss / total_num

def train_decoder_ood(backdoored_encoder, model, ood_data_loader, optimizer, epoch, args):
    """使用OOD数据训练解码器网络"""
    if ood_data_loader is None:
        return 0
    
    backdoored_encoder.eval()
    model.train()
    
    total_loss, total_num, train_bar = 0.0, 0, tqdm(ood_data_loader)
    for batch in train_bar:
        if isinstance(batch, list) or isinstance(batch, tuple):
            img = batch[0]
        else:
            img = batch
        z_img = 5 * torch.ones(img.shape).cuda()  # 标准化的目标图像
        img = img.cuda()
        with torch.no_grad():
            feature_raw = backdoored_encoder(img)
        predicted_img, mask = model(img, feature_raw)
        loss = torch.mean((predicted_img - z_img) ** 2 * mask) / args.mask_ratio
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * args.batch_size
        total_num += args.batch_size
        train_bar.set_description(f'Decoder Train Epoch: [{epoch}/{args.epochs}] OOD Loss: {total_loss / total_num:.4f}')

    return total_loss / total_num

def get_dataset_statistics(backdoored_encoder, model, dataset, args, save_cuda=False, test_mask_ratio=None):
    """获取数据集上的重建误差统计"""
    backdoored_encoder.eval()
    model.eval()
    
    # 如果提供了test_mask_ratio，临时修改模型的mask_ratio
    original_mask_ratio = None
    if test_mask_ratio is not None and hasattr(model.encoder, 'shuffle'):
        original_mask_ratio = model.encoder.shuffle.ratio
        model.encoder.shuffle.ratio = test_mask_ratio
        print(f"使用测试掩码比例: {test_mask_ratio} (训练时为 {original_mask_ratio})")
    
    if save_cuda is False:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, pin_memory=True, drop_last=False)
    else:
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, 
                                num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    errors = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing reconstruction errors"):
            if isinstance(batch, list) or isinstance(batch, tuple):
                val_img = batch[0].cuda()
            else:
                val_img = batch.cuda()

            with torch.no_grad():
                feature_raw = backdoored_encoder(val_img)
                predicted_val_img, mask = model(val_img, feature_raw)
            
            for i in range(val_img.shape[0]):
                error = torch.sum((val_img[i] - predicted_val_img[i]) ** 2 * mask[i]).item()
                errors.append(error)

    # 恢复原始mask_ratio
    if original_mask_ratio is not None:
        model.encoder.shuffle.ratio = original_mask_ratio

    return errors

def analyze_errors_and_get_threshold(clean_errors, poisoned_errors=None, outlier_percentile=95, original_paper_mode=False):
    """分析错误分布并确定阈值"""
    clean_errors = np.array(clean_errors)
    
    if original_paper_mode:
        # 原始论文的方法：使用训练集的均值乘以1.5作为阈值
        mean_error = np.mean(clean_errors)
        threshold = mean_error * 1.5
        print(f"使用原始论文方法: 错误均值 {mean_error:.4f} × 1.5 = 阈值 {threshold:.4f}")
        return threshold, None

    
    # 如果有毒样本错误可用，计算ROC统计
    if poisoned_errors is not None:
        poisoned_errors = np.array(poisoned_errors)
        all_errors = np.concatenate([clean_errors, poisoned_errors])
        all_labels = np.concatenate([np.zeros(len(clean_errors)), np.ones(len(poisoned_errors))])
        
        # 根据ROC曲线找到最佳阈值
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(all_labels, all_errors)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold, roc_auc
    
    # 默认方法：使用百分位数作为阈值
    threshold = np.percentile(clean_errors, outlier_percentile)

    return threshold, None

def sift_dataset(backdoored_encoder, model, dataset, threshold, args):
    """根据重建误差过滤数据集"""
    errors = get_dataset_statistics(backdoored_encoder, model, dataset, args, 
                                   save_cuda=True, test_mask_ratio=args.test_mask_ratio)
    clean_idxs = np.where(np.array(errors) < threshold)[0]
    poisoned_idxs = np.where(np.array(errors) >= threshold)[0]
    
    clean_dataset = Subset(dataset, clean_idxs)
    poisoned_dataset = Subset(dataset, poisoned_idxs)
    
    print(f"After filtering: kept {len(clean_idxs)} samples, removed {len(poisoned_idxs)} samples")
    return clean_dataset, poisoned_dataset, clean_idxs, poisoned_idxs

def save_data_to_csv(data_dict, filename):
    """将数据保存为CSV文件"""
    df = pd.DataFrame(data_dict)
    df.to_csv(filename, index=False)
    print(f"数据已保存到: {filename}")

def run_dede_detection(args, suspicious_model, suspicious_dataset, memory_dataset=None,
                      clean_test_dataset=None, poisoned_test_dataset=None):
    """
    运行DeDe后门检测算法
    
    参数:
        args: 配置参数
        suspicious_model: 可疑的自监督模型 (可能被后门攻击)
        suspicious_dataset: 训练数据集 (可能包含毒样本)，默认有2种选项：（1）预训练数据集的子集（2）指定的OOD数据集
        memory_dataset: 仅用于监测算法运行状态，默认是suspicious_dataset的干净版本
        clean_test_dataset: 干净的测试数据集 
        poisoned_test_dataset: 有毒的测试数据集 
    
    返回:
        result_dict: 包含检测结果的字典
    """
    start_time = time.time()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置matplotlib字体为Arial，字体大小为18
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['font.size'] = 18
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 将可疑模型设置为评估模式
    suspicious_model.eval()
    suspicious_model.cuda()
    
    # 创建DeDe解码器模型
    decoder_model = DecoderModel(
        image_size=args.image_size,
        patch_size=args.patch_size,
        emb_dim=args.emb_dim,
        encoder_layer=args.encoder_layer,
        encoder_head=args.encoder_head,
        decoder_layer=args.decoder_layer,
        decoder_head=args.decoder_head,
        mask_ratio=args.mask_ratio,
        arch=args.arch
    ).cuda()
    
    # 为解码器创建优化器，不对norm层应用weight decay
    decay, no_decay = [], []
    for name, param in decoder_model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过不需要梯度的参数
        # 根据参数名将参数分组，norm层、偏置、encoder.cls_token、encoder.pos_embedding 不应用weight decay
        if (
            name.endswith(".bias") or 
            len(param.shape) == 1 or 
            "norm" in name or 
            "ln" in name or 
            "LayerNorm" in name or 
            name.endswith(".cls_token") or 
            name.endswith(".pos_embedding")
        ):
            no_decay.append(param)
            print(f"不应用weight decay的参数: {name}")
        else:
            decay.append(param)
            print(f"应用weight decay的参数: {name}")
    
    optimizer_grouped_parameters = [
        {'params': decay, 'weight_decay': 1e-6},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    
    # 创建数据加载器
    memory_loader = DataLoader(
        memory_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    suspicious_loader = DataLoader(
        suspicious_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 初始化训练参数
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 如果需要恢复训练，加载checkpoint
    if hasattr(args, 'resume') and args.resume:
        checkpoint_path = args.checkpoint_path if hasattr(args, 'checkpoint_path') and args.checkpoint_path else None
        
        # 如果未指定checkpoint路径，自动查找最新的checkpoint
        if not checkpoint_path:
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
            if checkpoints:
                # 根据epoch数排序
                checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"自动找到最新的checkpoint: {checkpoint_path}")
            else:
                print("未找到可用的checkpoint，将从头开始训练")
        
        # 加载checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"正在从checkpoint恢复训练: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            decoder_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['loss']
            print(f"恢复训练从epoch {start_epoch}开始，当前最佳loss: {best_loss:.4f}")
        else:
            if checkpoint_path:
                print(f"指定的checkpoint路径不存在: {checkpoint_path}")
            print("将从头开始训练")
    
    # 训练解码器
    for epoch in range(start_epoch, args.epochs):
        # 在可疑数据上训练解码器
        train_loss = train_decoder(suspicious_model, decoder_model, suspicious_loader, optimizer, epoch, args)
        
        # 可选: 在OOD数据上训练解码器
        # if args.use_ood_training and suspicious_loader is not None:
        #     ood_loss = train_decoder_ood(suspicious_model, decoder_model, suspicious_loader, optimizer, epoch, args)
        
        # 保存当前模型checkpoint
        if hasattr(args, 'save_freq') and epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch:04d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"保存checkpoint到: {checkpoint_path}")
        
        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            best_checkpoint_path = os.path.join(args.output_dir, 'best_decoder.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, best_checkpoint_path)
            print(f"保存最佳模型到: {best_checkpoint_path}, loss: {best_loss:.4f}")
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_decoder.pth'))
    decoder_model.load_state_dict(checkpoint['model_state_dict'])

    print("Computing reconstruction errors on suspicious dataset...")
    suspicious_errors = get_dataset_statistics(suspicious_model, decoder_model, suspicious_dataset, args)
    
    # memory_errors 仅仅作为监测使用
    print("Computing reconstruction errors on memory dataset...")
    memory_errors = get_dataset_statistics(suspicious_model, decoder_model, memory_dataset, args)
    
    # 计算阈值
    threshold, _ = analyze_errors_and_get_threshold(suspicious_errors, original_paper_mode=True)
    
    # 过滤可疑数据集
    clean_dataset, poisoned_dataset, clean_idxs, poisoned_idxs = sift_dataset(
        suspicious_model, decoder_model, suspicious_dataset, threshold, args
    )
    
    # 保存过滤后的文件列表（如果是文件列表数据集）
    filtered_file_list = []
    if hasattr(suspicious_dataset, 'file_list'):
        for idx in clean_idxs:
            filtered_file_list.append(suspicious_dataset.file_list[idx])
        
        filtered_file_path = os.path.join(args.output_dir, 'filtered_file_list.txt')
        with open(filtered_file_path, 'w') as f:
            for line in filtered_file_list:
                f.write(f"{line}\n")
    else:
        print(f"Suspicious dataset does not have a file list, you should use a file list dataset")
    
    # 如果有测试数据集，评估过滤效果
    test_results = {}
    if clean_test_dataset is not None and poisoned_test_dataset is not None:
        print("Computing reconstruction errors on clean test dataset...")
        clean_test_errors = get_dataset_statistics(suspicious_model, decoder_model, clean_test_dataset, args, 
                                                 test_mask_ratio=args.test_mask_ratio)
        
        print("Computing reconstruction errors on poisoned test dataset...")
        poisoned_test_errors = get_dataset_statistics(suspicious_model, decoder_model, poisoned_test_dataset, args,
                                                   test_mask_ratio=args.test_mask_ratio)
        
        # 计算最佳阈值和ROC曲线
        optimal_threshold, roc_auc = analyze_errors_and_get_threshold(clean_test_errors, poisoned_test_errors)
        
        # 遵照原论文方法，计算基于clean_test_dataset误差均值的阈值(×1.5)
        test_threshold, _ = analyze_errors_and_get_threshold(clean_test_errors, poisoned_test_errors, original_paper_mode=True)
        
        
        # 计算使用最佳阈值的检测准确率
        clean_test_correct = sum(1 for e in clean_test_errors if e < optimal_threshold)
        poisoned_test_correct = sum(1 for e in poisoned_test_errors if e >= optimal_threshold)
        recall = poisoned_test_correct / len(poisoned_test_errors)
        precision = poisoned_test_correct / (poisoned_test_correct + sum(1 for e in clean_test_errors if e >= optimal_threshold))
        overall_acc = (clean_test_correct + poisoned_test_correct) / (len(clean_test_errors) + len(poisoned_test_errors))
        
        # 计算使用替代阈值的检测准确率
        alt_clean_test_correct = sum(1 for e in clean_test_errors if e < test_threshold)
        alt_poisoned_test_correct = sum(1 for e in poisoned_test_errors if e >= test_threshold)
        alt_recall = alt_poisoned_test_correct / len(poisoned_test_errors)
        alt_precision = alt_poisoned_test_correct / (alt_poisoned_test_correct + sum(1 for e in clean_test_errors if e >= test_threshold))
        alt_overall_acc = (alt_clean_test_correct + alt_poisoned_test_correct) / (len(clean_test_errors) + len(poisoned_test_errors))
        
        test_results = {
            'optimal_threshold': optimal_threshold,
            'roc_auc': roc_auc,
            'recall': recall,
            'precision': precision,
            'overall_accuracy': overall_acc,
            'test_threshold': test_threshold,
            'alt_recall': alt_recall,
            'alt_precision': alt_precision,
            'alt_overall_accuracy': alt_overall_acc
        }
        
        # 保存测试数据到CSV
        csv_data = {
            'clean_test_errors': clean_test_errors,
            'poisoned_test_errors': poisoned_test_errors
        }
        error_data_path = os.path.join(args.output_dir, 'test_error_data.csv')
        # 转换为DataFrame格式，处理不同长度的列
        max_len = max(len(clean_test_errors), len(poisoned_test_errors))
        csv_data = {
            'clean_test_errors': clean_test_errors + [np.nan] * (max_len - len(clean_test_errors)),
            'poisoned_test_errors': poisoned_test_errors + [np.nan] * (max_len - len(poisoned_test_errors))
        }
        save_data_to_csv(csv_data, error_data_path)
        
        # 绘制重建误差分布，使用测试集数据
        plt.figure(figsize=(10, 6))
        plt.hist(clean_test_errors, bins=50, alpha=0.5, label='Clean Test', density=True)
        plt.hist(poisoned_test_errors, bins=50, alpha=0.5, label='Poisoned Test', density=True)
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
        plt.axvline(x=test_threshold, color='green', linestyle='-.', label=f'1.5xTrainError: {test_threshold:.2f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        # 不要title
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'error_distribution_test.png'))
    
    # 保存训练数据到CSV
    csv_data = {
        'memory_errors': memory_errors,
        'suspicious_errors': suspicious_errors
    }
    # 转换为DataFrame格式，处理不同长度的列
    max_len = max(len(memory_errors), len(suspicious_errors))
    csv_data = {
        'memory_errors': memory_errors + [np.nan] * (max_len - len(memory_errors)),
        'suspicious_errors': suspicious_errors + [np.nan] * (max_len - len(suspicious_errors))
    }
    error_data_path = os.path.join(args.output_dir, 'training_error_data.csv')
    save_data_to_csv(csv_data, error_data_path)
    
    # 绘制重建误差分布
    plt.figure(figsize=(10, 6))
    plt.hist(memory_errors, bins=50, alpha=0.5, label='Memory (Clean)', density=True)
    plt.hist(suspicious_errors, bins=50, alpha=0.5, label='Suspicious', density=True)
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'error_distribution.png'))
    
    # 准备结果字典
    elapsed_time = time.time() - start_time
    result_dict = {
        'threshold': threshold,
        'clean_indices': clean_idxs.tolist(),
        'poisoned_indices': poisoned_idxs.tolist(),
        'clean_set_size': len(clean_idxs),
        'poisoned_set_size': len(poisoned_idxs),
        'memory_errors_mean': np.mean(memory_errors),
        'memory_errors_std': np.std(memory_errors),
        'suspicious_errors_mean': np.mean(suspicious_errors),
        'suspicious_errors_std': np.std(suspicious_errors),
        'elapsed_time': elapsed_time,
        'test_results': test_results
    }
    
    # 保存结果
    with open(os.path.join(args.output_dir, 'dede_results.pkl'), 'wb') as f:
        pickle.dump(result_dict, f)
    
    print(f"DeDe detection completed in {elapsed_time:.2f} seconds")
    print(f"Found {len(poisoned_idxs)} potential poisoned samples")
    print(f"Results saved to {args.output_dir}")
    
    return result_dict, clean_dataset, poisoned_dataset 