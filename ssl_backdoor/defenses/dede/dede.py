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

# 新增: 将图像反归一化→调整尺寸→重新归一化，以便当 shadow_size 与 DeDe image_size 不同时仍能使用
def _resize_for_dede(img: torch.Tensor, target_size: int, mean, std):
    """将 Normalize 后的张量先反归一化、Resize，再重新归一化。

    参数:
        img: Tensor(B,C,H,W) —— 已标准化
        target_size: resize 目标边长
        mean, std: 序列 or Tensor，数据集实际使用的归一化参数
    """
    if img.shape[-1] == target_size:
        return img

    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, device=img.device)
    if not torch.is_tensor(std):
        std = torch.tensor(std, device=img.device)
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)

    img_denorm = img * std + mean
    img_resized = F.interpolate(img_denorm, size=(target_size, target_size), mode="bilinear", align_corners=False)
    img_norm = (img_resized - mean) / std
    return img_norm

# 递归提取 Normalize 的均值方差
from torchvision import transforms

def _extract_mean_std(transform):
    if isinstance(transform, transforms.Normalize):
        return transform.mean, transform.std
    elif hasattr(transform, 'transforms'):
        for t in transform.transforms:
            res = _extract_mean_std(t)
            if res is not None:
                return res
    # Add support for SkipAugmentationForTensor wrapper (check full_transform)
    elif hasattr(transform, 'full_transform'):
        return _extract_mean_std(transform.full_transform)
    return None

from ssl_backdoor.utils.utils import set_seed

    
def train_decoder(backdoored_encoder, model, data_loader, optimizer, epoch, args):
    """训练解码器网络。支持梯度累积:`args.accum_steps` 个 micro-batch 累积一次梯度后再 step。"""
    backdoored_encoder.eval()
    model.train()

    accum_steps = max(1, int(getattr(args, 'accum_steps', 1)))
    grad_clip = float(getattr(args, 'max_grad_norm', 1.0))

    optimizer.zero_grad()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    step = 0
    for step, batch in enumerate(train_bar):
        if isinstance(batch, list) or isinstance(batch, tuple):
            img = batch[0]
        else:
            img = batch

        img = img.cuda()
        with torch.no_grad():
            feature_raw = backdoored_encoder(img)
        decoder_input = _resize_for_dede(img, args.image_size, args.mean, args.std)
        predicted_img, mask = model(decoder_input, feature_raw)
        loss = torch.mean((predicted_img - decoder_input) ** 2 * mask) / args.mask_ratio
        (loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * args.batch_size
        total_num += args.batch_size
        train_bar.set_description(f'Decoder Train Epoch: [{epoch}/{args.epochs}] Loss: {total_loss / total_num:.4f}')

    # 处理结尾不满一个累积窗口的残余梯度
    if (step + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

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
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing reconstruction errors"):
                if isinstance(batch, list) or isinstance(batch, tuple):
                    val_img = batch[0].cuda()
                else:
                    val_img = batch.cuda()

                with torch.no_grad():
                    feature_raw = backdoored_encoder(val_img)

                    decoder_input = _resize_for_dede(val_img, args.image_size, args.mean, args.std)

                    predicted_val_img, mask = model(decoder_input, feature_raw)

                for i in range(val_img.shape[0]):
                    error = torch.sum((decoder_input[i] - predicted_val_img[i]) ** 2 * mask[i]).item()
                    errors.append(error)
    finally:
        # 即便循环中抛出异常,也必须恢复原始 mask_ratio,否则会污染后续训练
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
                                   save_cuda=True, test_mask_ratio=getattr(args, "test_mask_ratio", None))
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

def run_dede_detection(args, suspicious_model, suspicious_dataset,
                      clean_test_dataset=None, poisoned_test_dataset=None,
                      suspicious_dataset_gt=None):
    """
    运行DeDe后门检测算法
    
    参数:
        args: 配置参数
        suspicious_model: 可疑的自监督模型 (可能被后门攻击)
        suspicious_dataset: 训练数据集 (可能包含毒样本)，默认有2种选项：（1）预训练数据集的子集（2）指定的OOD数据集
        clean_test_dataset: 干净的测试数据集 
        poisoned_test_dataset: 有毒的测试数据集 
        suspicious_dataset_gt: 训练数据集的 ground truth 标签 (0为干净, 1为有毒)，可选
    
    返回:
        result_dict: 包含检测结果的字典
    """
    start_time = time.time()
    
    # 设置随机种子
    set_seed(args.seed)

    # ========= 提取归一化参数 =========
    if hasattr(suspicious_dataset, 'transform') and suspicious_dataset.transform is not None:
        ms = _extract_mean_std(suspicious_dataset.transform)
    else:
        ms = None

    if ms is None:
        # 若外部已在 args 中提供 mean/std（例如来自 HuggingFace processor），则优先使用
        if hasattr(args, 'mean') and hasattr(args, 'std') and args.mean is not None and args.std is not None:
            print(f"使用预设 mean/std: {args.mean}, {args.std}")
        else:
            # 否则回退到 ImageNet 默认
            default_mean = [0.485, 0.456, 0.406]
            default_std = [0.229, 0.224, 0.225]
            args.mean, args.std = default_mean, default_std
            print(f"未能解析Normalize，使用默认 ImageNet mean/std: {args.mean}, {args.std}")
    else:
        args.mean, args.std = list(ms[0]), list(ms[1])
        print(f"已解析Normalize mean/std: {args.mean}, {args.std}")
    
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

    # 当 backbone 输出维度未知(arch 含 'clip'/'siglip' 或未识别)时,DecoderModel.mlp 是 LazyLinear,
    # 此时 param.shape 为 UninitializedParameter,后续按参数分组遍历会报错。先做一次 dummy forward
    # 触发 LazyLinear 的真实初始化。
    with torch.no_grad():
        _dummy_img = torch.zeros(1, 3, args.image_size, args.image_size, device='cuda')
        _dummy_feat = suspicious_model(_dummy_img)
        decoder_model(_dummy_img, _dummy_feat)

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

    # save_strategy:'latest' 仅保留每 epoch 覆盖的 latest_decoder.pth(+ best);
    # 其他值(或未设)沿用旧行为 —— 按 save_freq 写 checkpoint_<epoch>.pth。
    save_strategy = str(getattr(args, 'save_strategy', '')).lower()
    eval_freq = int(getattr(args, 'eval_freq', 0) or 0)
    eval_test_mask_ratio = getattr(args, 'test_mask_ratio', None)
    eval_csv_path = os.path.join(args.output_dir, 'eval_curve.csv')
    if eval_freq > 0 and (clean_test_dataset is None or poisoned_test_dataset is None):
        print("[warn] eval_freq>0 但缺 clean/poisoned test dataset,跳过周期评估。")
        eval_freq = 0
    if eval_freq > 0 and not os.path.exists(eval_csv_path):
        with open(eval_csv_path, 'w') as f:
            f.write('epoch,train_loss,clean_err_mean,poison_err_mean,clean_err_median,poison_err_median,roc_auc\n')

    # best AUROC 跟踪 —— 与 best_loss 完全独立, 仅在 eval_freq>0 时启用
    best_auroc = -float('inf')
    best_auroc_path = os.path.join(args.output_dir, 'best_auroc_decoder.pth')

    # resume 场景下,best_loss / best_auroc 不能依赖 latest checkpoint 的字段,
    # 否则历史 best 可能被更差的 latest 覆盖。直接从历史 best 文件回读基准值。
    best_decoder_path = os.path.join(args.output_dir, 'best_decoder.pth')
    if start_epoch > 0 and os.path.exists(best_decoder_path):
        try:
            prev_best = torch.load(best_decoder_path, map_location='cpu')
            prev_best_loss = float(prev_best.get('loss', float('inf')))
            if prev_best_loss < best_loss:
                best_loss = prev_best_loss
            print(f"[resume] 历史 best_decoder.pth loss={prev_best_loss:.4f}, 取 min => best_loss={best_loss:.4f}")
        except Exception as _e:
            print(f"[resume] 读取历史 best_decoder.pth 失败: {_e}")
    if start_epoch > 0 and os.path.exists(best_auroc_path):
        try:
            prev_best_auroc = torch.load(best_auroc_path, map_location='cpu')
            prev_auroc = float(prev_best_auroc.get('roc_auc', -float('inf')))
            if prev_auroc > best_auroc:
                best_auroc = prev_auroc
            print(f"[resume] 历史 best_auroc_decoder.pth AUROC={prev_auroc:.4f}, 取 max => best_auroc={best_auroc:.4f}")
        except Exception as _e:
            print(f"[resume] 读取历史 best_auroc_decoder.pth 失败: {_e}")

    # 训练解码器
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_decoder(suspicious_model, decoder_model, suspicious_loader, optimizer, epoch, args)

        # ===== checkpoint 保存 =====
        if save_strategy == 'latest':
            latest_path = os.path.join(args.output_dir, 'latest_decoder.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, latest_path)
        elif hasattr(args, 'save_freq') and (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch:04d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"保存checkpoint到: {checkpoint_path}")

        # 保存最佳模型(始终保留,推理依赖)
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

        # ===== 训练中周期评估(消融用),用 test_mask_ratio 算 clean/poison AUROC =====
        if eval_freq > 0 and ((epoch + 1) % eval_freq == 0 or (epoch + 1) == args.epochs):
            try:
                clean_errs = get_dataset_statistics(suspicious_model, decoder_model,
                                                   clean_test_dataset, args,
                                                   test_mask_ratio=eval_test_mask_ratio)
                poison_errs = get_dataset_statistics(suspicious_model, decoder_model,
                                                    poisoned_test_dataset, args,
                                                    test_mask_ratio=eval_test_mask_ratio)
                from sklearn.metrics import roc_auc_score
                y_true = np.concatenate([np.zeros(len(clean_errs)), np.ones(len(poison_errs))])
                y_score = np.concatenate([clean_errs, poison_errs])
                roc_auc = float(roc_auc_score(y_true, y_score))
                with open(eval_csv_path, 'a') as f:
                    f.write(f'{epoch},{train_loss:.6f},{np.mean(clean_errs):.6f},'
                            f'{np.mean(poison_errs):.6f},{np.median(clean_errs):.6f},'
                            f'{np.median(poison_errs):.6f},{roc_auc:.6f}\n')
                print(f"[eval@ep{epoch}] AUROC={roc_auc:.4f} | "
                      f"clean μ={np.mean(clean_errs):.4f} | poison μ={np.mean(poison_errs):.4f}")
                if roc_auc > best_auroc:
                    best_auroc = roc_auc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': decoder_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        'roc_auc': roc_auc,
                    }, best_auroc_path)
                    print(f"保存最佳AUROC模型到: {best_auroc_path}, AUROC: {best_auroc:.4f}")
            except Exception as _e:
                print(f"[eval@ep{epoch}] 失败: {_e}")
            finally:
                # 无论 eval 成功与否, 都要把 decoder 切回 train 模式, 避免下一个 epoch 训练在 eval 模式下进行
                decoder_model.train()
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_decoder.pth'))
    decoder_model.load_state_dict(checkpoint['model_state_dict'])

    print("Computing reconstruction errors on suspicious dataset...")
    suspicious_errors = get_dataset_statistics(suspicious_model, decoder_model, suspicious_dataset, args)
    
    # 计算阈值
    threshold, _ = analyze_errors_and_get_threshold(suspicious_errors, original_paper_mode=True)
    
    # 过滤可疑数据集
    clean_dataset, poisoned_dataset, clean_idxs, poisoned_idxs = sift_dataset(
        suspicious_model, decoder_model, suspicious_dataset, threshold, args
    )
    
    # 如果提供了训练集的 ground truth，计算检测指标
    train_results = {}
    if suspicious_dataset_gt is not None:
        print("Computing detection metrics on suspicious (training) dataset...")
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            suspicious_dataset_gt = np.array(suspicious_dataset_gt)
            scores = np.array(suspicious_errors)
            
            # 确保长度一致
            if len(suspicious_dataset_gt) == len(scores):
                # 计算 AUROC, AUPRC
                # 注意：如果全为0或全为1，roc_auc_score会报错，需要处理
                if len(np.unique(suspicious_dataset_gt)) > 1:
                    roc_auc = roc_auc_score(suspicious_dataset_gt, scores)
                    auprc = average_precision_score(suspicious_dataset_gt, scores)
                else:
                    roc_auc = 0.5
                    auprc = 0.0
                    print("Warning: suspicious_dataset_gt contains only one class.")
                
                # 计算 TPR, FPR, Precision based on threshold
                predicted_labels = np.zeros_like(suspicious_dataset_gt)
                predicted_labels[poisoned_idxs] = 1
                
                tp = np.sum((predicted_labels == 1) & (suspicious_dataset_gt == 1))
                fp = np.sum((predicted_labels == 1) & (suspicious_dataset_gt == 0))
                tn = np.sum((predicted_labels == 0) & (suspicious_dataset_gt == 0))
                fn = np.sum((predicted_labels == 0) & (suspicious_dataset_gt == 1))
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                train_results = {
                    'roc_auc': roc_auc,
                    'auprc': auprc,
                    'tpr': tpr,
                    'fpr': fpr,
                    'precision': precision,
                    'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
                }
                print(f"Train set detection results: AUROC={roc_auc:.4f}, AUPRC={auprc:.4f}")
                print(f"TPR={tpr:.4f}, FPR={fpr:.4f}, Precision={precision:.4f}")
            else:
                print(f"Warning: Length mismatch. GT: {len(suspicious_dataset_gt)}, Errors: {len(scores)}")
        except Exception as e:
            print(f"Error computing train metrics: {e}")

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

        # 计算 AUPRC
        from sklearn.metrics import average_precision_score
        all_test_errors = np.concatenate([clean_test_errors, poisoned_test_errors])
        all_test_labels = np.concatenate([np.zeros(len(clean_test_errors)), np.ones(len(poisoned_test_errors))])
        auprc = average_precision_score(all_test_labels, all_test_errors)
        
        # 遵照原论文方法，计算基于clean_test_dataset误差均值的阈值(×1.5)
        test_threshold, _ = analyze_errors_and_get_threshold(clean_test_errors, poisoned_test_errors, original_paper_mode=True)
        
        
        # 计算使用最佳阈值的检测准确率
        clean_test_correct = sum(1 for e in clean_test_errors if e < optimal_threshold)
        poisoned_test_correct = sum(1 for e in poisoned_test_errors if e >= optimal_threshold)
        recall = poisoned_test_correct / len(poisoned_test_errors)
        tpr = recall
        fpr = (len(clean_test_errors) - clean_test_correct) / len(clean_test_errors)
        precision = poisoned_test_correct / (poisoned_test_correct + sum(1 for e in clean_test_errors if e >= optimal_threshold))
        overall_acc = (clean_test_correct + poisoned_test_correct) / (len(clean_test_errors) + len(poisoned_test_errors))
        
        # 计算使用替代阈值的检测准确率
        alt_clean_test_correct = sum(1 for e in clean_test_errors if e < test_threshold)
        alt_poisoned_test_correct = sum(1 for e in poisoned_test_errors if e >= test_threshold)
        alt_recall = alt_poisoned_test_correct / len(poisoned_test_errors)
        alt_tpr = alt_recall
        alt_fpr = (len(clean_test_errors) - alt_clean_test_correct) / len(clean_test_errors)
        alt_precision = alt_poisoned_test_correct / (alt_poisoned_test_correct + sum(1 for e in clean_test_errors if e >= test_threshold))
        alt_overall_acc = (alt_clean_test_correct + alt_poisoned_test_correct) / (len(clean_test_errors) + len(poisoned_test_errors))
        
        test_results = {
            'optimal_threshold': optimal_threshold,
            'roc_auc': roc_auc,
            'auprc': auprc,
            'recall': recall,
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision,
            'overall_accuracy': overall_acc,
            'test_threshold': test_threshold,
            'alt_recall': alt_recall,
            'alt_tpr': alt_tpr,
            'alt_fpr': alt_fpr,
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
        plt.axvline(x=test_threshold, color='green', linestyle='-.', label=f'1.5xCleanTestError: {test_threshold:.2f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        # 不要title
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'error_distribution_test.png'))
    
    # 保存训练数据到CSV（仅保存可疑数据集误差）
    csv_data = {'suspicious_errors': suspicious_errors}
    error_data_path = os.path.join(args.output_dir, 'training_error_data.csv')
    save_data_to_csv(csv_data, error_data_path)
    
    # 绘制重建误差分布
    plt.figure(figsize=(10, 6))
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
        'suspicious_errors_mean': np.mean(suspicious_errors),
        'suspicious_errors_std': np.std(suspicious_errors),
        'elapsed_time': elapsed_time,
        'test_results': test_results,
        'train_results': train_results
    }
    
    # 保存结果
    with open(os.path.join(args.output_dir, 'dede_results.pkl'), 'wb') as f:
        pickle.dump(result_dict, f)
    
    print(f"DeDe detection completed in {elapsed_time:.2f} seconds")
    print(f"Found {len(poisoned_idxs)} potential poisoned samples")
    print(f"Results saved to {args.output_dir}")
    
    return result_dict, clean_dataset, poisoned_dataset 