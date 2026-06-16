import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
import random
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

from .utils import epsilon, assert_range, compute_self_cos_sim, dump_img, set_seed, generate_mask
from ssl_backdoor.datasets import dataset_params

logger = logging.getLogger(__name__)

def _to_pixel_hwc(batch, device=None):
    """将图像 batch 统一为 DECREE 期望的 [B,H,W,C], float32, 值域 [0,255]。

    兼容两种常见来源，避免训练 / 目标特征 / 评估三条路径各自处理导致的不一致：
      - 形状 [B,C,H,W] 或 [B,H,W,C]，自动转成 HWC；
      - 值域 [0,1]（ToTensor 后未乘 255）或 [0,255]，自动统一到 [0,255]。

    DECREE 的触发器在原始像素空间（[0,255]）上构造，因此不接受已经 Normalize
    （去均值）的输入；若检测到明显负值则直接报错，避免按错误尺度静默算出无效指标。
    """
    if not torch.is_tensor(batch):
        raise TypeError("DECREE 期望张量输入")
    if batch.dim() != 4 or 3 not in (batch.shape[1], batch.shape[-1]):
        raise ValueError(f"无法识别的图像 batch 形状: {tuple(batch.shape)}，应为 [B,C,H,W] 或 [B,H,W,C]")
    if batch.shape[1] == 3 and batch.shape[-1] != 3:
        batch = batch.permute(0, 2, 3, 1)
    if device is not None:
        batch = batch.to(device)
    batch = batch.to(dtype=torch.float32)
    if float(batch.min().item()) < -1e-3:
        raise ValueError("检测到负值像素：DECREE 需要原始像素（[0,1] 或 [0,255]），不支持已 Normalize 的输入")
    if float(batch.max().item()) <= 1.0 + 1e-3:
        batch = batch * 255.0
    return batch


def _get_decree_input_size(args) -> int:
    """
    DECREE 输入/触发器尺寸（一个超参数同时决定：输入分辨率 + 触发器掩码/补丁尺寸）。
    优先从 args 获取，否则基于 dataset_id 从 dataset_params 自动推断。
    """
    if hasattr(args, 'decree_input_size') and getattr(args, 'decree_input_size') is not None:
        return int(getattr(args, 'decree_input_size'))
    
    # 自动推断
    dataset_id = _get_decree_dataset_id(args)
    if dataset_id in dataset_params:
        return dataset_params[dataset_id].get('image_size', 224) # 默认 224
    
    raise ValueError(f"无法推断 input_size: 未提供 decree_input_size 且数据集 '{dataset_id}' 不在 dataset_params 中")


def _get_decree_dataset_id(args) -> str:
    """
    数据集标识（用于归一化统计量选择、结果命名等）。
    """
    if not hasattr(args, 'decree_dataset_id') or getattr(args, 'decree_dataset_id') is None:
        # 如果没有明确提供 decree_dataset_id，尝试从 args.dataset 获取
        if hasattr(args, 'dataset') and getattr(args, 'dataset') is not None:
            return str(getattr(args, 'dataset'))
        raise ValueError("缺少 decree_dataset_id（可选：imagenet/cifar10/stl10 等）")
    
    dataset_id = str(getattr(args, 'decree_dataset_id'))
    if dataset_id not in dataset_params:
        raise ValueError(f"不支持的 decree_dataset_id: {dataset_id}（可选：{list(dataset_params.keys())}）")
    return dataset_id


def _get_decree_lambda_min(args, input_size: int) -> float:
    """
    与旧实现一致：224 分支使用固定的 1e-7；其余尺寸（含 32/96 等）使用 args.lambda_min。
    """
    if input_size == 224:
        return 1e-7
    return float(getattr(args, 'lambda_min'))


def _trigger_inv_dir(args, succ_threshold: float, lambda_min: float) -> str:
    dataset_id = _get_decree_dataset_id(args)
    input_size = _get_decree_input_size(args)
    return os.path.join(
        args.output_dir,
        f'trigger_inv/d{dataset_id}_s{input_size}_{succ_threshold}_{lambda_min}_{args.seed}_{args.batch_size}_{args.lr}_{args.mask_init}'
    )


def adjust_learning_rate(optimizer, epoch, args):
    """根据训练轮次调整学习率"""
    input_size = _get_decree_input_size(args)
    thres = {224: [200, 500], 32: [30, 50]}.get(input_size)
    if thres is None:
        # 非预置尺寸（如 stl10 的 96）：按总轮次比例回退一个调度，并告警提示其未经调参。
        thres = [int(args.epochs * 0.5), int(args.epochs * 0.83)]
        logger.warning(f"decree_input_size={input_size} 无预置 LR 调度，按 epochs 比例回退到里程碑 {thres}")

    if epoch < thres[0]:
        lr = args.lr
    elif epoch < thres[1]:
        lr = 0.1
    else:
        lr = 0.05
    
    logger.info(f'轮次: {epoch}  学习率: {lr:.4f}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def decree_detector(args, suspicious_model, clean_train_loader):
    """
    DECREE检测算法核心实现
    
    参数:
        args: 算法参数
        suspicious_model: 可疑模型，必须由用户提前加载
        clean_train_loader: 干净数据的DataLoader实例
    
    返回:
        regular_best: 最佳L1范数
        duration: 运行时间
        res_best: 最佳触发器掩码和补丁
        target_backdoor_feature: 目标后门特征（在干净样本上应用触发器后提取的平均特征）
    """
    # 检查模型是否有效
    if suspicious_model is None:
        raise ValueError("必须提供已加载的模型(suspicious_model)，DECREE不再自动加载模型")
    
    # 1. 设置随机种子
    set_seed(args.seed)
    
    # 设置设备：优先使用 CUDA（由 CUDA_VISIBLE_DEVICES 控制可见设备），否则回退到 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 设置模型到正确设备
    model = suspicious_model.to(device)
    
    # 3. 确定触发器几何参数（仅用于初始化；不再依赖任何 trigger 先验文件）
    mask_size = _get_decree_input_size(args)
    _trigger_geom = {224: (24, 24, 176), 32: (5, 5, 22)}
    if mask_size in _trigger_geom:
        trigger_h, trigger_w, trigger_r = _trigger_geom[mask_size]
    else:
        # 非预置尺寸：按边距≈10% 比例回退几何（触发器区域 r = size - 2*offset），并告警
        trigger_h = trigger_w = max(1, round(mask_size * 0.1))
        trigger_r = mask_size - 2 * trigger_h
        logger.warning(f"decree_input_size={mask_size} 无预置触发器几何，按比例回退到 offset={trigger_h}, r={trigger_r}")
    
    logger.info(f'使用模型: {args.weights_path}')
    
    # 4. 初始化触发器（不再加载任何 trigger 文件；只按 mask_size 初始化形状）
    logger.info(f'掩码尺寸: {mask_size}')
    
    # 根据初始化方法设置掩码和补丁
    if args.mask_init == 'orc':  # 使用原始掩码形状
        mask, patch = generate_mask(mask_size, trigger_h, trigger_w, r=trigger_r)
        train_mask_2d = torch.tensor(mask, dtype=torch.float64).to(device)
        train_patch = torch.rand((mask_size, mask_size, 3), dtype=torch.float64).to(device)
    elif args.mask_init == 'rand':  # 随机初始化
        train_mask_2d = torch.rand((mask_size, mask_size), dtype=torch.float64).to(device)
        train_patch = torch.rand((mask_size, mask_size, 3), dtype=torch.float64).to(device)
    else:
        raise ValueError(f"不支持的掩码初始化方法: {args.mask_init}")
    
    # 对掩码和补丁进行变换，确保值域在(0,1)
    train_mask_2d = torch.arctanh((train_mask_2d - 0.5) * (2 - epsilon()))
    train_patch = torch.arctanh((train_patch - 0.5) * (2 - epsilon()))
    train_mask_2d.requires_grad = True
    train_patch.requires_grad = True
    
    # 5. 准备数据集和数据加载器
    dataset_id = _get_decree_dataset_id(args)
    test_transform = transforms.Compose([
        dataset_params[dataset_id]['normalize']
    ])
    
    logger.info(f'使用数据集 {dataset_id}，影子数据集变换: {test_transform}')
    
    # 6. 设置优化器和优化目标
    projectee = torch.rand([1, 512], dtype=torch.float64).to(device)
    projectee = F.normalize(projectee, dim=-1)
    optimizer = torch.optim.Adam(params=[train_mask_2d, train_patch],
                                lr=args.lr, betas=(0.5, 0.9))
    
    # 模型设置为评估模式
    model.eval()
    
    # 7. 开始优化触发器
    loss_cos, loss_reg = None, None
    # lambda 的初始化值会在计算出 lambda_min 后再确定，保证不会低于 lambda_min
    init_loss_lambda = None
    loss_lambda = None  # 平衡余弦损失和正则化损失的权重
    adaptor_lambda = 5.0  # 动态调整lambda权重的系数
    patience = 5
    succ_threshold = args.thres  # 成功反转触发器的余弦相似度阈值
    epochs = 1000
    
    # 早停机制
    regular_best = 1 / epsilon()
    early_stop_reg_best = regular_best
    early_stop_cnt = 0
    
    # 调整lambda值的参数
    adaptor_up_cnt, adaptor_down_cnt = 0, 0
    adaptor_up_flag, adaptor_down_flag = False, False
    lambda_set_cnt = 0
    
    # 根据不同数据集设置不同参数
    input_size = _get_decree_input_size(args)
    lambda_min = _get_decree_lambda_min(args, input_size)
    lambda_set_patience = 2 * patience
    # 224 用更长的早停耐心；32 及其余尺寸沿用较短默认值
    early_stop_patience = (7 if input_size == 224 else 2) * patience

    # 初始化 loss_lambda：默认从 1e-3 开始，但不会低于 lambda_min
    init_loss_lambda = max(1e-3, float(lambda_min))
    loss_lambda = init_loss_lambda
    
    logger.info(f'配置参数: lambda_min: {lambda_min}, '
               f'adapt_lambda: {adaptor_lambda}, '
               f'lambda_set_patience: {lambda_set_patience}, '
               f'succ_threshold: {succ_threshold}, '
               f'early_stop_patience: {early_stop_patience}')
    
    regular_list, cosine_list = [], []
    start_time = time.time()
    
    # 用于保存找到的最佳结果
    res_best = {'mask': None, 'patch': None}
    
    # 用于保存目标后门特征
    target_backdoor_feature = None

    def _compute_target_backdoor_feature_from_best_trigger(max_batches: int = 5):
        """
        使用当前找到的最佳触发器（res_best）在少量干净样本上生成目标后门特征。

        注意：
        - 只依赖 res_best 的 mask/patch，不依赖当前 epoch/step 的 loss 变量，
          避免出现“特征用的是旧触发器/随机触发器”的情况。
        - 输入 clean_x_batch 既可能是 [B,H,W,C]（0-255）也可能是 [B,C,H,W]；
          若是 float 且 max<=1，则按 [0,1] 解释并放缩到 [0,255]。
        """
        if res_best.get('mask') is None or res_best.get('patch') is None:
            return None

        bd_features_collection = []
        with torch.no_grad():
            for step, (clean_x_batch, _) in enumerate(clean_train_loader):
                if step >= int(max_batches):
                    break

                # 统一到 [B,H,W,C], float32, 值域 [0,255]
                clean_x_batch = _to_pixel_hwc(clean_x_batch, device)

                mask = res_best['mask'].to(device=device, dtype=torch.float32)
                patch = res_best['patch'].to(device=device, dtype=torch.float32)

                bd_x_batch = (1 - mask) * clean_x_batch + mask * patch
                bd_x_batch = torch.clip(bd_x_batch, min=0, max=255)

                bd_input = []
                for i in range(bd_x_batch.shape[0]):
                    bd_trans = test_transform(bd_x_batch[i].permute(2, 0, 1) / 255.0)
                    bd_input.append(bd_trans)

                if not bd_input:
                    continue

                bd_input = torch.stack(bd_input).to(dtype=torch.float).to(device)
                bd_out = model(bd_input)
                bd_features_collection.append(bd_out.detach())

        if not bd_features_collection:
            return None

        all_bd_features = torch.cat(bd_features_collection, dim=0)
        curr_target_feature = torch.mean(all_bd_features, dim=0, keepdim=True)
        curr_target_feature = F.normalize(curr_target_feature, dim=-1)
        return curr_target_feature

    for e in range(epochs):
        # 调整学习率
        adjust_learning_rate(optimizer, e, args)
        
        loss_best = {'loss': [], 'cos': [], 'reg': []}
        max_clean_l1 = 0
        
        # 逐批次处理数据
        for step, (clean_x_batch, _) in enumerate(clean_train_loader):
            # 统一到 [B,H,W,C], float32, 值域 [0,255]（兼容 CHW / [0,1] 输入）
            clean_x_batch = _to_pixel_hwc(clean_x_batch, device)

            # 统计clean_x_batch在[0,1]空间中的L1范数最大值
            clean_x_batch_01 = clean_x_batch / 255.0
            l1s = clean_x_batch_01.abs().view(clean_x_batch_01.shape[0], -1).sum(dim=1)
            batch_max = l1s.max().item()
            if batch_max > max_clean_l1:
                max_clean_l1 = batch_max
            
            # 生成掩码和补丁
            train_mask_3d = train_mask_2d.unsqueeze(2).repeat(1, 1, 3)  # 扩展为3D
            train_mask_tanh = torch.tanh(train_mask_3d) / (2 - epsilon()) + 0.5  # 范围(0,1)
            train_patch_tanh = (torch.tanh(train_patch) / (2 - epsilon()) + 0.5) * 255  # 范围(0,255)
            train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
            train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)
            
            # 生成后门图像
            bd_x_batch = (1 - train_mask_tanh) * clean_x_batch + \
                         train_mask_tanh * train_patch_tanh
            bd_x_batch = torch.clip(bd_x_batch, min=0, max=255)
            
            # 准备模型输入
            clean_input, bd_input = [], []
            for i in range(clean_x_batch.shape[0]):
                clean_trans = test_transform(clean_x_batch[i].permute(2, 0, 1) / 255.0)
                bd_trans = test_transform(bd_x_batch[i].permute(2, 0, 1) / 255.0)
                clean_input.append(clean_trans)
                bd_input.append(bd_trans)
            
            clean_input = torch.stack(clean_input)
            bd_input = torch.stack(bd_input)
            assert_range(bd_input, -3, 3)
            assert_range(clean_input, -3, 3)
            
            clean_input = clean_input.to(dtype=torch.float).to(device)
            bd_input = bd_input.to(dtype=torch.float).to(device)
            
            # 前向传播
            bd_out = model(bd_input)
            
            # 计算损失
            loss_cos = (-compute_self_cos_sim(bd_out))
            loss_reg = torch.sum(torch.abs(train_mask_tanh))  # L1正则化
            loss = loss_cos + loss_reg * loss_lambda
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 收集损失信息
            loss_best['loss'].append(loss.item())
            loss_best['cos'].append(loss_cos.item())
            loss_best['reg'].append(loss_reg.item())
            
            # 更新最佳结果
            if (torch.abs(loss_cos) > succ_threshold) and (loss_reg < regular_best):
                train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
                train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)
                res_best['mask'] = train_mask_tanh.detach()
                res_best['patch'] = train_patch_tanh.detach()
                regular_best = loss_reg
            
            # 检查早停条件
            if regular_best < 1 / epsilon():  # 已找到有效触发器
                if regular_best >= early_stop_reg_best:
                    early_stop_cnt += 1
                else:
                    early_stop_cnt = 0
            early_stop_reg_best = min(regular_best, early_stop_reg_best)
            
            # 调整lambda值
            if loss_lambda < lambda_min and (torch.abs(loss_cos) > succ_threshold):
                lambda_set_cnt += 1
                if lambda_set_cnt > lambda_set_patience:
                    loss_lambda = init_loss_lambda
                    adaptor_up_cnt, adaptor_down_cnt = 0, 0
                    adaptor_up_flag, adaptor_down_flag = False, False
                    logger.info(f"初始化lambda值为 {loss_lambda}")
            else:
                lambda_set_cnt = 0
            
            if (torch.abs(loss_cos) > succ_threshold):
                adaptor_up_cnt += 1
                adaptor_down_cnt = 0
            else:
                adaptor_down_cnt += 1
                adaptor_up_cnt = 0
            
            if (adaptor_up_cnt > patience):
                if loss_lambda < 1e5:
                    loss_lambda *= adaptor_lambda
                adaptor_up_cnt = 0
                adaptor_up_flag = True
                logger.info(f'步骤{step}: lambda值上调至 {loss_lambda}')
            elif (adaptor_down_cnt > patience):
                # 旧实现会在 loss_lambda == lambda_min 时也做一次除法，导致低于最小值（例如 1e-3 / 5 = 2e-4）。
                # 这里强制把 loss_lambda 夹在 [lambda_min, +inf)。
                if loss_lambda > lambda_min:
                    loss_lambda = max(loss_lambda / adaptor_lambda, float(lambda_min))
                adaptor_down_cnt = 0
                adaptor_down_flag = True
                logger.info(f'步骤{step}: lambda值下调至 {loss_lambda}')
        
        # 计算本轮平均损失
        loss_avg_e = np.mean(loss_best['loss'])
        loss_cos_e = np.mean(loss_best['cos'])
        loss_reg_e = np.mean(loss_best['reg'])
        
        logger.info(f"轮次={e}, 损失={loss_avg_e:.6f}, 余弦损失={loss_cos_e:.6f}, "
                   f"正则化损失={loss_reg_e:.6f}, 当前最佳L1={regular_best:.6f}, "
                   f"早停最佳L1={early_stop_reg_best:.6f}")
        logger.info(f"[0,1]空间clean_x_batch L1范数最大值: {max_clean_l1:.4f}")
        
        regular_list.append(str(round(float(loss_reg_e), 2)))
        cosine_list.append(str(round(float(-loss_cos_e), 2)))
        
        # 如果有最佳结果，保存触发器
        if res_best['mask'] is not None and res_best['patch'] is not None:
            assert_range(res_best['mask'], 0, 1)
            assert_range(res_best['patch'], 0, 255)
            
            fusion = np.asarray((res_best['mask'] * res_best['patch']).detach().cpu(), np.uint8)
            mask = np.asarray(res_best['mask'].detach().cpu() * 255, np.uint8)
            patch = np.asarray(res_best['patch'].detach().cpu(), np.uint8)
            
            # 创建保存目录
            trigger_dir = _trigger_inv_dir(args, succ_threshold, lambda_min)
            os.makedirs(trigger_dir, exist_ok=True)
            
            # 保存触发器图像
            suffix = f'e{e}_reg{regular_best:.2f}'
            Image.fromarray(mask).save(f'{trigger_dir}/mask_{suffix}.png')
            Image.fromarray(patch).save(f'{trigger_dir}/patch_{suffix}.png')
            Image.fromarray(fusion).save(f'{trigger_dir}/fus_{suffix}.png')
        
        # 早停检查
        if abs(loss_cos_e) > succ_threshold and early_stop_cnt > early_stop_patience:
            logger.info('达到早停条件，停止训练!')
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f'用时: {duration:.4f}秒')
            logger.info(f'最终L1范数: {regular_best:.4f}')
            logger.info(f"正则化损失历史: {','.join(regular_list)}")
            logger.info(f"余弦相似度历史: {','.join(cosine_list)}")               
            # 使用最终 best trigger 生成目标后门特征（用于后续 poisoned 检测指标）
            target_backdoor_feature = _compute_target_backdoor_feature_from_best_trigger(max_batches=5)
            if target_backdoor_feature is not None:
                logger.info(f"生成目标后门特征完成，形状: {target_backdoor_feature.shape}")
            return regular_best, duration, res_best, target_backdoor_feature
    
    # 如果没有早停，返回最终结果
    duration = time.time() - start_time

    # 使用最终 best trigger 生成目标后门特征（用于后续 poisoned 检测指标）
    target_backdoor_feature = _compute_target_backdoor_feature_from_best_trigger(max_batches=5)
    if target_backdoor_feature is not None:
        logger.info(f"生成目标后门特征完成，形状: {target_backdoor_feature.shape}")

    return regular_best, duration, res_best, target_backdoor_feature

def run_decree_detection(args, suspicious_model=None, suspicious_dataset=None,
                         clean_test_dataset=None, poisoned_test_dataset=None,
                         detect_clean_dataset=None, detect_poisoned_dataset=None,
                         detect_dataset=None):
    """
    运行DECREE后门检测算法
    
    参数:
        args: 配置参数
        suspicious_model: 可疑的自监督模型 (必须由用户提前加载)
        suspicious_dataset: 训练数据集 (可能包含毒样本)，由调用者提供
        clean_test_dataset: 干净的测试数据集，用于评估分类性能
        poisoned_test_dataset: 带毒的测试数据集，用于评估分类性能
        detect_clean_dataset: 待检测数据集（干净版），用于 poisoned image 检测指标评估
        detect_poisoned_dataset: 待检测数据集（带毒版），用于 poisoned image 检测指标评估
        detect_dataset: 待检测数据集（单一混合列表），用 img_path 是否包含 'poison' 作为真值标签
    
    返回:
        result_dict: 包含检测结果的字典
    """
    # 检查是否提供了模型和数据集
    if suspicious_model is None:
        raise ValueError("必须提供已加载的模型(suspicious_model)，DECREE不再自动加载模型")
    if suspicious_dataset is None:
        raise ValueError("必须提供已加载的数据集(suspicious_dataset)")
        
    start_time = time.time()
    
    # 设置输出目录
    det_log_dir = os.path.join(args.output_dir, 'detect_log')
    os.makedirs(det_log_dir, exist_ok=True)
    
    # 设置结果文件
    result_file = os.path.join(args.output_dir, 'decree_results.txt')

    
    # 配置logger
    file_handler = logging.FileHandler(os.path.join(det_log_dir, f'decree_{args.seed}_lr{args.lr}_b{args.batch_size}_{args.mask_init}.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 创建DataLoader
    logger.info(f'使用提供的数据集进行DECREE检测，数据集大小: {len(suspicious_dataset)}')
    clean_train_loader = DataLoader(suspicious_dataset,
                                   batch_size=args.batch_size,
                                   pin_memory=True,
                                   shuffle=True)

    # 运行DECREE检测
    # decree_detector 现在返回4个值，包括目标后门特征
    l1_norm, duration, res_best_trigger, target_backdoor_feature = decree_detector(args, suspicious_model, clean_train_loader)
    
    # 解析结果
    results = {
        'encoder_path': args.weights_path,
        'l1_norm': l1_norm,
        'duration': duration
    }
    
    # 检查是否成功生成了触发器反转结果
    # 注意：与 decree_detector 内部保存目录的 lambda_min 规则保持一致（224->1e-7；32->args.lambda_min）
    input_size = _get_decree_input_size(args)
    lambda_min_for_dir = _get_decree_lambda_min(args, input_size)
    trigger_inv_dir = _trigger_inv_dir(args, args.thres, lambda_min_for_dir)
    if os.path.exists(trigger_inv_dir):
        results['trigger_inv_dir'] = trigger_inv_dir
        # 获取最后一个生成的触发器（按epoch排序）
        trigger_files = [f for f in os.listdir(trigger_inv_dir) if f.startswith('fus_')]
        if trigger_files:
            results['trigger_files'] = sorted(trigger_files)
    
    elapsed_time = time.time() - start_time
    results['elapsed_time'] = elapsed_time
    
    # 写入结果文件
    with open(result_file, 'a') as f:
        f.write(f"{args.weights_path},{l1_norm:.4f},{duration:.4f}\n")
    
    logger.info(f"DECREE检测完成，耗时 {elapsed_time:.2f} 秒")
    if 'l1_norm' in results:
        logger.info(f"L1范数: {results['l1_norm']:.4f}")

    def _select_test_transform():
        dataset_id = _get_decree_dataset_id(args)
        return transforms.Compose([
            dataset_params[dataset_id]['normalize']
        ])

    def _compute_auprc(scores, labels):
        """
        计算 AUPRC（Average Precision / PR 曲线面积的常见离散近似）。
        - scores: list[float]
        - labels: list[int]  (1=poisoned, 0=clean)
        """
        if not scores or not labels:
            return 0.0
        total_pos = int(sum(labels))
        if total_pos <= 0:
            return 0.0
        sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
        tp = 0
        fp = 0
        prev_recall = 0.0
        auprc = 0.0
        for _, lab in sorted_pairs:
            if int(lab) == 1:
                tp += 1
            else:
                fp += 1
            recall = tp / total_pos
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            auprc += (recall - prev_recall) * precision
            prev_recall = recall
        return float(auprc)

    def _evaluate_poison_detection(clean_ds, poisoned_ds, log_prefix, similarity_threshold):
        """
        在给定 clean/poisoned 数据集上，计算 poisoned image 检测指标。
        输出字段（满足你的需求）：
          - tpr, fpr, recall, precision, auroc, auprc
        同时保留原有字段：roc_auc、specified_*、optimal_*（如可计算）。
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_transform = _select_test_transform()

        clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        poisoned_loader = DataLoader(poisoned_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        all_scores = []
        all_labels = []
        clean_scores = []
        poison_scores = []

        suspicious_model.eval()
        with torch.no_grad():
            logger.info(f"{log_prefix}正在评估干净样本...")
            for x_batch, _ in clean_loader:
                # 统一到 [B,H,W,C], float32, 值域 [0,255]（兼容 CHW / [0,1] 输入）
                x_batch = _to_pixel_hwc(x_batch, device)

                input_list = []
                for i in range(x_batch.shape[0]):
                    img_chw_01 = x_batch[i].permute(2, 0, 1) / 255.0
                    input_list.append(test_transform(img_chw_01))
                if not input_list:
                    continue
                feats = suspicious_model(torch.stack(input_list).to(device))
                feats = F.normalize(feats, dim=1)
                sims = F.cosine_similarity(feats, target_backdoor_feature.repeat(feats.size(0), 1))
                all_scores.extend(sims.cpu().tolist())
                clean_scores.extend(sims.cpu().tolist())
                all_labels.extend([0] * feats.size(0))

            logger.info(f"{log_prefix}正在评估带毒样本...")
            for x_batch, _ in poisoned_loader:
                # 统一到 [B,H,W,C], float32, 值域 [0,255]（兼容 CHW / [0,1] 输入）
                x_batch = _to_pixel_hwc(x_batch, device)

                input_list = []
                for i in range(x_batch.shape[0]):
                    img_chw_01 = x_batch[i].permute(2, 0, 1) / 255.0
                    input_list.append(test_transform(img_chw_01))
                if not input_list:
                    continue
                feats = suspicious_model(torch.stack(input_list).to(device))
                feats = F.normalize(feats, dim=1)
                sims = F.cosine_similarity(feats, target_backdoor_feature.repeat(feats.size(0), 1))
                all_scores.extend(sims.cpu().tolist())
                poison_scores.extend(sims.cpu().tolist())
                all_labels.extend([1] * feats.size(0))

        if not all_scores or not all_labels:
            return {"error": "No scores or labels generated."}

        metrics = {}

        # --- debug stats: mean similarity for clean/poison ---
        if clean_scores:
            metrics['mean_sim_clean'] = float(np.mean(clean_scores))
        if poison_scores:
            metrics['mean_sim_poison'] = float(np.mean(poison_scores))
        if clean_scores and poison_scores:
            logger.info(
                f"{log_prefix}相似度均值: clean={metrics['mean_sim_clean']:.6f}, "
                f"poison={metrics['mean_sim_poison']:.6f}"
            )

        # --- AUROC ---
        sorted_pairs = sorted(zip(all_scores, all_labels), key=lambda x: x[0], reverse=True)
        sorted_scores, sorted_labels = zip(*sorted_pairs)
        total_positive = int(sum(sorted_labels))
        total_negative = int(len(sorted_labels) - total_positive)

        tpr_list = []
        fpr_list = []
        thresholds = []
        tp = 0
        fp = 0
        last_score = float('inf')
        for score, label in sorted_pairs:
            if score != last_score:
                tpr_list.append(tp / total_positive if total_positive > 0 else 0.0)
                fpr_list.append(fp / total_negative if total_negative > 0 else 0.0)
                thresholds.append(score)
                last_score = score
            if int(label) == 1:
                tp += 1
            else:
                fp += 1
        tpr_list.append(tp / total_positive if total_positive > 0 else 0.0)
        fpr_list.append(fp / total_negative if total_negative > 0 else 0.0)

        auc = 0.0
        for i in range(1, len(tpr_list)):
            auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
        metrics['roc_auc'] = float(auc)
        metrics['auroc'] = float(auc)

        # --- AUPRC ---
        auprc = _compute_auprc(all_scores, all_labels)
        metrics['auprc'] = float(auprc)

        # --- optimal threshold (Youden's J) ---
        if thresholds:
            best_j = -1.0
            best_threshold_idx = 0
            for i in range(len(thresholds)):
                j = tpr_list[i] - fpr_list[i]
                if j > best_j:
                    best_j = j
                    best_threshold_idx = i
            optimal_threshold = thresholds[best_threshold_idx]
            metrics['optimal_threshold'] = float(optimal_threshold)

            tp_opt = fp_opt = tn_opt = fn_opt = 0
            for score, label in zip(all_scores, all_labels):
                pred = 1 if score >= optimal_threshold else 0
                if pred == 1 and label == 1:
                    tp_opt += 1
                elif pred == 1 and label == 0:
                    fp_opt += 1
                elif pred == 0 and label == 0:
                    tn_opt += 1
                elif pred == 0 and label == 1:
                    fn_opt += 1
            metrics['optimal_precision'] = tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0.0
            metrics['optimal_recall'] = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0.0
            metrics['optimal_f1'] = 2 * metrics['optimal_precision'] * metrics['optimal_recall'] / (metrics['optimal_precision'] + metrics['optimal_recall']) if (metrics['optimal_precision'] + metrics['optimal_recall']) > 0 else 0.0

        # --- specified threshold metrics (poisoned detection) ---
        tp_spec = fp_spec = tn_spec = fn_spec = 0
        for score, label in zip(all_scores, all_labels):
            pred = 1 if score >= similarity_threshold else 0
            if pred == 1 and label == 1:
                tp_spec += 1
            elif pred == 1 and label == 0:
                fp_spec += 1
            elif pred == 0 and label == 0:
                tn_spec += 1
            elif pred == 0 and label == 1:
                fn_spec += 1

        precision = tp_spec / (tp_spec + fp_spec) if (tp_spec + fp_spec) > 0 else 0.0
        recall = tp_spec / (tp_spec + fn_spec) if (tp_spec + fn_spec) > 0 else 0.0
        tpr = recall
        fpr = fp_spec / (fp_spec + tn_spec) if (fp_spec + tn_spec) > 0 else 0.0

        metrics['specified_threshold'] = float(similarity_threshold)
        metrics['specified_precision'] = float(precision)
        metrics['specified_recall'] = float(recall)
        metrics['specified_f1'] = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # 你要求的字段（在指定阈值下）
        metrics['threshold'] = float(similarity_threshold)
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['tpr'] = float(tpr)
        metrics['fpr'] = float(fpr)

        return metrics

    def _evaluate_mixed_dataset_detection(mixed_ds, log_prefix, similarity_threshold, poison_keyword="poison"):
        """
        单一混合数据集评估：用样本路径是否包含 poison_keyword 作为 poisoned 真值标签。
        需要 mixed_ds 返回 rich_output（包含 img_path 字段）。
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_transform = _select_test_transform()

        loader = DataLoader(mixed_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        all_scores = []
        all_labels = []

        suspicious_model.eval()
        with torch.no_grad():
            logger.info(f"{log_prefix}正在评估混合数据集样本（poison 关键字: '{poison_keyword}'）...")
            for batch in loader:
                if isinstance(batch, dict):
                    x_batch = batch.get('img')
                    paths = batch.get('img_path')
                else:
                    # 没有路径就无法生成 poison 真值标签
                    return {"error": "Mixed dataset must provide rich_output with img_path."}

                if x_batch is None or paths is None:
                    return {"error": "Mixed dataset batch missing img or img_path."}

                # 统一到 [B,H,W,C], float32, 值域 [0,255]（兼容 CHW / [0,1] 输入）
                x_batch = _to_pixel_hwc(x_batch, device)

                input_list = []
                for i in range(x_batch.shape[0]):
                    img_chw_01 = x_batch[i].permute(2, 0, 1) / 255.0
                    input_list.append(test_transform(img_chw_01))

                if not input_list:
                    continue

                feats = suspicious_model(torch.stack(input_list).to(device))
                feats = F.normalize(feats, dim=1)
                sims = F.cosine_similarity(feats, target_backdoor_feature.repeat(feats.size(0), 1))

                # paths: list[str]
                labels = [1 if (poison_keyword in str(p)) else 0 for p in paths]
                sims_list = sims.cpu().tolist()
                all_scores.extend(sims_list)
                all_labels.extend(labels)

        if not all_scores or not all_labels:
            return {"error": "No scores or labels generated."}

        # 复用二分类评估：把 labels 当成 (0=clean,1=poisoned)
        # 这里直接走 _evaluate_poison_detection 的内部实现不方便复用，因此复制其统计段
        metrics = {}

        # --- debug stats: mean similarity for clean/poison ---
        if all_labels:
            clean_vals = [s for s, l in zip(all_scores, all_labels) if int(l) == 0]
            poison_vals = [s for s, l in zip(all_scores, all_labels) if int(l) == 1]
            if clean_vals:
                metrics['mean_sim_clean'] = float(np.mean(clean_vals))
            if poison_vals:
                metrics['mean_sim_poison'] = float(np.mean(poison_vals))
            if clean_vals and poison_vals:
                logger.info(
                    f"{log_prefix}相似度均值: clean={metrics['mean_sim_clean']:.6f}, "
                    f"poison={metrics['mean_sim_poison']:.6f}"
                )

        sorted_pairs = sorted(zip(all_scores, all_labels), key=lambda x: x[0], reverse=True)
        total_positive = int(sum(all_labels))
        total_negative = int(len(all_labels) - total_positive)

        # AUROC
        tpr_list = []
        fpr_list = []
        thresholds = []
        tp = 0
        fp = 0
        last_score = float('inf')
        for score, label in sorted_pairs:
            if score != last_score:
                tpr_list.append(tp / total_positive if total_positive > 0 else 0.0)
                fpr_list.append(fp / total_negative if total_negative > 0 else 0.0)
                thresholds.append(score)
                last_score = score
            if int(label) == 1:
                tp += 1
            else:
                fp += 1
        tpr_list.append(tp / total_positive if total_positive > 0 else 0.0)
        fpr_list.append(fp / total_negative if total_negative > 0 else 0.0)

        auc = 0.0
        for i in range(1, len(tpr_list)):
            auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
        metrics['roc_auc'] = float(auc)
        metrics['auroc'] = float(auc)

        # AUPRC
        metrics['auprc'] = float(_compute_auprc(all_scores, all_labels))

        # specified threshold (你要的 TPR/FPR/Recall/Precision)
        tp_spec = fp_spec = tn_spec = fn_spec = 0
        for score, label in zip(all_scores, all_labels):
            pred = 1 if score >= similarity_threshold else 0
            if pred == 1 and label == 1:
                tp_spec += 1
            elif pred == 1 and label == 0:
                fp_spec += 1
            elif pred == 0 and label == 0:
                tn_spec += 1
            elif pred == 0 and label == 1:
                fn_spec += 1

        precision = tp_spec / (tp_spec + fp_spec) if (tp_spec + fp_spec) > 0 else 0.0
        recall = tp_spec / (tp_spec + fn_spec) if (tp_spec + fn_spec) > 0 else 0.0
        fpr = fp_spec / (fp_spec + tn_spec) if (fp_spec + tn_spec) > 0 else 0.0

        metrics['specified_threshold'] = float(similarity_threshold)
        metrics['threshold'] = float(similarity_threshold)
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['tpr'] = float(recall)
        metrics['fpr'] = float(fpr)

        # 记录样本量，方便 sanity check
        metrics['n_total'] = int(len(all_labels))
        metrics['n_poisoned'] = int(total_positive)
        metrics['n_clean'] = int(total_negative)

        return metrics

    # 基于触发器相似度的 poisoned 检测指标评估（支持两套数据：旧 test 数据 + 新 detect 数据）
    can_eval = (
        res_best_trigger and res_best_trigger.get('mask') is not None and res_best_trigger.get('patch') is not None and
        target_backdoor_feature is not None
    )
    if can_eval:
        similarity_threshold = getattr(args, 'similarity_eval_threshold', 0.9)
        logger.info(f"\n====== 开始基于触发器的 poisoned 检测指标评估 (阈值={similarity_threshold:.4f}) ======")

        # 旧逻辑：test_config 指定的测试集（如果提供）
        if clean_test_dataset is not None and poisoned_test_dataset is not None:
            metrics = _evaluate_poison_detection(
                clean_test_dataset, poisoned_test_dataset,
                log_prefix="[test_dataset] ",
                similarity_threshold=similarity_threshold
            )
            results['sample_classification_metrics'] = metrics

            if "error" not in metrics:
                logger.info(f"[test_dataset] poisoned 检测指标: AUROC={metrics.get('auroc', 0.0):.4f}, AUPRC={metrics.get('auprc', 0.0):.4f}, "
                            f"TPR={metrics.get('tpr', 0.0):.4f}, FPR={metrics.get('fpr', 0.0):.4f}, "
                            f"Precision={metrics.get('precision', 0.0):.4f}, Recall={metrics.get('recall', 0.0):.4f}")
        else:
            logger.info("未提供 test clean/poisoned 数据集，跳过 test_dataset 指标。")

        # 新增逻辑：config 指定的待检测数据集（单文件混合列表）
        if detect_dataset is not None:
            detect_metrics = _evaluate_mixed_dataset_detection(
                detect_dataset,
                log_prefix="[detect_dataset] ",
                similarity_threshold=similarity_threshold,
                poison_keyword="poison"
            )
            results['detect_dataset_metrics'] = detect_metrics
            if "error" not in detect_metrics:
                logger.info(f"[detect_dataset] poisoned 检测指标: AUROC={detect_metrics.get('auroc', 0.0):.4f}, AUPRC={detect_metrics.get('auprc', 0.0):.4f}, "
                            f"TPR={detect_metrics.get('tpr', 0.0):.4f}, FPR={detect_metrics.get('fpr', 0.0):.4f}, "
                            f"Precision={detect_metrics.get('precision', 0.0):.4f}, Recall={detect_metrics.get('recall', 0.0):.4f} "
                            f"(N={detect_metrics.get('n_total', 'N/A')}, P={detect_metrics.get('n_poisoned', 'N/A')}, N0={detect_metrics.get('n_clean', 'N/A')})")
        else:
            logger.info("未提供 detect_dataset（单文件混合列表），跳过 detect_dataset 指标。")
    else:
        if target_backdoor_feature is None:
            logger.info("未生成目标后门特征，跳过 poisoned 检测指标评估。")
    
    # 移除添加的handler，避免重复记录
    logger.removeHandler(file_handler)
    
    return results 
