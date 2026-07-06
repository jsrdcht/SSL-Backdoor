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

def _to_pixel_hwc(batch, device=None):    """Helper function."""
    if not torch.is_tensor(batch):
        raise TypeError("DECREE expects a tensor input")
    if batch.dim() != 4 or 3 not in (batch.shape[1], batch.shape[-1]):
        raise ValueError(f"Cannot infer image batch shape: {tuple(batch.shape)}. Expected [B, C, H, W] or [B, H, W, C]")
    if batch.shape[1] == 3 and batch.shape[-1] != 3:
        batch = batch.permute(0, 2, 3, 1)
    if device is not None:
        batch = batch.to(device)
    batch = batch.to(dtype=torch.float32)
    if float(batch.min().item()) < -1e-3:
        raise ValueError("Detected negative pixel values. DECREE expects raw pixels in [0, 1] or [0, 255], normalized inputs are not supported")
    if float(batch.max().item()) <= 1.0 + 1e-3:
        batch = batch * 255.0
    return batch


def _get_decree_input_size(args) -> int:    """Helper function."""
    if hasattr(args, 'decree_input_size') and getattr(args, 'decree_input_size') is not None:
        return int(getattr(args, 'decree_input_size'))
    
    
    dataset_id = _get_decree_dataset_id(args)
    if dataset_id in dataset_params:
        return dataset_params[dataset_id].get('image_size', 224) 
    
    raise ValueError(f"Cannot infer input_size: decree_input_size is missing and dataset '{dataset_id}' is not in dataset_params")


def _get_decree_dataset_id(args) -> str:    """Helper function."""
    if not hasattr(args, 'decree_dataset_id') or getattr(args, 'decree_dataset_id') is None:
        
        if hasattr(args, 'dataset') and getattr(args, 'dataset') is not None:
            return str(getattr(args, 'dataset'))
        raise ValueError("Missing decree_dataset_id (optional values include imagenet, cifar10, stl10, etc.)")
    
    dataset_id = str(getattr(args, 'decree_dataset_id'))
    if dataset_id not in dataset_params:
        raise ValueError(f"Unsupported decree_dataset_id: {dataset_id} (supported: {list(dataset_params.keys())})")
    return dataset_id


def _get_decree_lambda_min(args, input_size: int) -> float:    """Helper function."""
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
    """Helper function.."""
    input_size = _get_decree_input_size(args)
    thres = {224: [200, 500], 32: [30, 50]}.get(input_size)
    if thres is None:
        
        thres = [int(args.epochs * 0.5), int(args.epochs * 0.83)]
        logger.warning(f"decree_input_size={input_size} has no preset LR schedule; fallback to epoch-ratio milestones {thres}")

    if epoch < thres[0]:
        lr = args.lr
    elif epoch < thres[1]:
        lr = 0.1
    else:
        lr = 0.05
    
    logger.info(f'Epoch: {epoch}, learning rate: {lr:.4f}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def decree_detector(args, suspicious_model, clean_train_loader):
    """Helper function.
        Helper function.
    
        Helper function.
        Helper function.
        Helper function.
        Helper function.
    
        Helper function.
        Helper function.
        Helper function.
        Helper function.
        Helper function.
    """
    
    if suspicious_model is None:
        raise ValueError("A loaded suspicious_model is required. DECREE does not auto-load the model anymore")
    
    
    set_seed(args.seed)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model = suspicious_model.to(device)
    
    
    mask_size = _get_decree_input_size(args)
    _trigger_geom = {224: (24, 24, 176), 32: (5, 5, 22)}
    if mask_size in _trigger_geom:
        trigger_h, trigger_w, trigger_r = _trigger_geom[mask_size]
    else:
        
        trigger_h = trigger_w = max(1, round(mask_size * 0.1))
        trigger_r = mask_size - 2 * trigger_h
        logger.warning(f"decree_input_size={mask_size} has no preset trigger geometry; fallback to proportional values offset={trigger_h}, r={trigger_r}")
    
    logger.info(f'Using model: {args.weights_path}')
    
    
    logger.info(f'Mask size: {mask_size}')
    
    
    if args.mask_init == 'orc':  
        mask, patch = generate_mask(mask_size, trigger_h, trigger_w, r=trigger_r)
        train_mask_2d = torch.tensor(mask, dtype=torch.float64).to(device)
        train_patch = torch.rand((mask_size, mask_size, 3), dtype=torch.float64).to(device)
    elif args.mask_init == 'rand':  
        train_mask_2d = torch.rand((mask_size, mask_size), dtype=torch.float64).to(device)
        train_patch = torch.rand((mask_size, mask_size, 3), dtype=torch.float64).to(device)
    else:
        raise ValueError(f"Unsupported mask initialization method: {args.mask_init}")
    
    
    train_mask_2d = torch.arctanh((train_mask_2d - 0.5) * (2 - epsilon()))
    train_patch = torch.arctanh((train_patch - 0.5) * (2 - epsilon()))
    train_mask_2d.requires_grad = True
    train_patch.requires_grad = True
    
    
    dataset_id = _get_decree_dataset_id(args)
    test_transform = transforms.Compose([
        dataset_params[dataset_id]['normalize']
    ])
    
    logger.info(f'Using dataset {dataset_id}, shadow transform: {test_transform}')
    
    
    projectee = torch.rand([1, 512], dtype=torch.float64).to(device)
    projectee = F.normalize(projectee, dim=-1)
    optimizer = torch.optim.Adam(params=[train_mask_2d, train_patch],
                                lr=args.lr, betas=(0.5, 0.9))
    
    
    model.eval()
    
    
    loss_cos, loss_reg = None, None
    
    init_loss_lambda = None
    loss_lambda = None  
    adaptor_lambda = 5.0  
    patience = 5
    succ_threshold = args.thres  
    epochs = 1000
    
    
    regular_best = 1 / epsilon()
    early_stop_reg_best = regular_best
    early_stop_cnt = 0
    
    
    adaptor_up_cnt, adaptor_down_cnt = 0, 0
    adaptor_up_flag, adaptor_down_flag = False, False
    lambda_set_cnt = 0
    
    
    input_size = _get_decree_input_size(args)
    lambda_min = _get_decree_lambda_min(args, input_size)
    lambda_set_patience = 2 * patience
    
    early_stop_patience = (7 if input_size == 224 else 2) * patience

    
    init_loss_lambda = max(1e-3, float(lambda_min))
    loss_lambda = init_loss_lambda
    
    logger.info(f'Configuration: lambda_min: {lambda_min}, '
               f'adapt_lambda: {adaptor_lambda}, '
               f'lambda_set_patience: {lambda_set_patience}, '
               f'succ_threshold: {succ_threshold}, '
               f'early_stop_patience: {early_stop_patience}')
    
    regular_list, cosine_list = [], []
    start_time = time.time()
    
    
    res_best = {'mask': None, 'patch': None}
    
    
    target_backdoor_feature = None

    def _compute_target_backdoor_feature_from_best_trigger(max_batches: int = 5):        """Helper function."""
        if res_best.get('mask') is None or res_best.get('patch') is None:
            return None

        bd_features_collection = []
        with torch.no_grad():
            for step, (clean_x_batch, _) in enumerate(clean_train_loader):
                if step >= int(max_batches):
                    break

                
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
        
        adjust_learning_rate(optimizer, e, args)
        
        loss_best = {'loss': [], 'cos': [], 'reg': []}
        max_clean_l1 = 0
        
        
        for step, (clean_x_batch, _) in enumerate(clean_train_loader):
            
            clean_x_batch = _to_pixel_hwc(clean_x_batch, device)

            
            clean_x_batch_01 = clean_x_batch / 255.0
            l1s = clean_x_batch_01.abs().view(clean_x_batch_01.shape[0], -1).sum(dim=1)
            batch_max = l1s.max().item()
            if batch_max > max_clean_l1:
                max_clean_l1 = batch_max
            
            
            train_mask_3d = train_mask_2d.unsqueeze(2).repeat(1, 1, 3)  
            train_mask_tanh = torch.tanh(train_mask_3d) / (2 - epsilon()) + 0.5  
            train_patch_tanh = (torch.tanh(train_patch) / (2 - epsilon()) + 0.5) * 255  
            train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
            train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)
            
            
            bd_x_batch = (1 - train_mask_tanh) * clean_x_batch +\
                         train_mask_tanh * train_patch_tanh
            bd_x_batch = torch.clip(bd_x_batch, min=0, max=255)
            
            
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
            
            
            bd_out = model(bd_input)
            
            
            loss_cos = (-compute_self_cos_sim(bd_out))
            loss_reg = torch.sum(torch.abs(train_mask_tanh))  
            loss = loss_cos + loss_reg * loss_lambda
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            loss_best['loss'].append(loss.item())
            loss_best['cos'].append(loss_cos.item())
            loss_best['reg'].append(loss_reg.item())
            
            
            if (torch.abs(loss_cos) > succ_threshold) and (loss_reg < regular_best):
                train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
                train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)
                res_best['mask'] = train_mask_tanh.detach()
                res_best['patch'] = train_patch_tanh.detach()
                regular_best = loss_reg
            
            
            if regular_best < 1 / epsilon():  
                if regular_best >= early_stop_reg_best:
                    early_stop_cnt += 1
                else:
                    early_stop_cnt = 0
            early_stop_reg_best = min(regular_best, early_stop_reg_best)
            
            
            if loss_lambda < lambda_min and (torch.abs(loss_cos) > succ_threshold):
                lambda_set_cnt += 1
                if lambda_set_cnt > lambda_set_patience:
                    loss_lambda = init_loss_lambda
                    adaptor_up_cnt, adaptor_down_cnt = 0, 0
                    adaptor_up_flag, adaptor_down_flag = False, False
                    logger.info(f"Initialized lambda to {loss_lambda}")
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
                logger.info(f'Step {step}: lambda increased to {loss_lambda}')
            elif (adaptor_down_cnt > patience):
                
                
                if loss_lambda > lambda_min:
                    loss_lambda = max(loss_lambda / adaptor_lambda, float(lambda_min))
                adaptor_down_cnt = 0
                adaptor_down_flag = True
                logger.info(f'Step {step}: lambda decreased to {loss_lambda}')
        
        
        loss_avg_e = np.mean(loss_best['loss'])
        loss_cos_e = np.mean(loss_best['cos'])
        loss_reg_e = np.mean(loss_best['reg'])
        
        logger.info(f"Epoch={e}, loss={loss_avg_e:.6f}, cosine_loss={loss_cos_e:.6f}, "
                   f"regularization_loss={loss_reg_e:.6f}, best_L1={regular_best:.6f}, "
                   f"early_stop_best_L1={early_stop_reg_best:.6f}")
        logger.info(f"Max L1 norm in [0,1] clean_x_batch: {max_clean_l1:.4f}")
        
        regular_list.append(str(round(float(loss_reg_e), 2)))
        cosine_list.append(str(round(float(-loss_cos_e), 2)))
        
        
        if res_best['mask'] is not None and res_best['patch'] is not None:
            assert_range(res_best['mask'], 0, 1)
            assert_range(res_best['patch'], 0, 255)
            
            fusion = np.asarray((res_best['mask'] * res_best['patch']).detach().cpu(), np.uint8)
            mask = np.asarray(res_best['mask'].detach().cpu() * 255, np.uint8)
            patch = np.asarray(res_best['patch'].detach().cpu(), np.uint8)
            
            
            trigger_dir = _trigger_inv_dir(args, succ_threshold, lambda_min)
            os.makedirs(trigger_dir, exist_ok=True)
            
            
            suffix = f'e{e}_reg{regular_best:.2f}'
            Image.fromarray(mask).save(f'{trigger_dir}/mask_{suffix}.png')
            Image.fromarray(patch).save(f'{trigger_dir}/patch_{suffix}.png')
            Image.fromarray(fusion).save(f'{trigger_dir}/fus_{suffix}.png')
        
        
        if abs(loss_cos_e) > succ_threshold and early_stop_cnt > early_stop_patience:
            logger.info('Early stopping condition met, stopping training.')
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f'Elapsed time: {duration:.4f}s')
            logger.info(f'Final L1 norm: {regular_best:.4f}')
            logger.info(f"Regularization loss history: {','.join(regular_list)}")
            logger.info(f"Cosine similarity history: {','.join(cosine_list)}")               
            
            target_backdoor_feature = _compute_target_backdoor_feature_from_best_trigger(max_batches=5)
            if target_backdoor_feature is not None:
                logger.info(f"Target backdoor feature generation completed, shape: {target_backdoor_feature.shape}")
            return regular_best, duration, res_best, target_backdoor_feature
    
    
    duration = time.time() - start_time

    
    target_backdoor_feature = _compute_target_backdoor_feature_from_best_trigger(max_batches=5)
    if target_backdoor_feature is not None:
        logger.info(f"Target backdoor feature generation completed, shape: {target_backdoor_feature.shape}")

    return regular_best, duration, res_best, target_backdoor_feature

def run_decree_detection(args, suspicious_model=None, suspicious_dataset=None,
                         clean_test_dataset=None, poisoned_test_dataset=None,
                         detect_clean_dataset=None, detect_poisoned_dataset=None,
                         detect_dataset=None):
    """Helper function.
        Helper function.
    
        Helper function.
        Helper function.
        Helper function.
        Helper function.
        Helper function.
        Helper function.
        Helper function.
        Helper function.
        Helper function.
    
        Helper function.
        Helper function.
    """
    
    if suspicious_model is None:
        raise ValueError("A loaded suspicious_model is required. DECREE does not auto-load the model anymore")
    if suspicious_dataset is None:
        raise ValueError("A loaded suspicious_dataset is required")
        
    start_time = time.time()
    
    
    det_log_dir = os.path.join(args.output_dir, 'detect_log')
    os.makedirs(det_log_dir, exist_ok=True)
    
    
    result_file = os.path.join(args.output_dir, 'decree_results.txt')

    
    
    file_handler = logging.FileHandler(os.path.join(det_log_dir, f'decree_{args.seed}_lr{args.lr}_b{args.batch_size}_{args.mask_init}.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    
    logger.info(f'Running DECREE with provided dataset, dataset size: {len(suspicious_dataset)}')
    clean_train_loader = DataLoader(suspicious_dataset,
                                   batch_size=args.batch_size,
                                   pin_memory=True,
                                   shuffle=True)

    
    
    l1_norm, duration, res_best_trigger, target_backdoor_feature = decree_detector(args, suspicious_model, clean_train_loader)
    
    
    results = {
        'encoder_path': args.weights_path,
        'l1_norm': l1_norm,
        'duration': duration
    }
    
    
    
    input_size = _get_decree_input_size(args)
    lambda_min_for_dir = _get_decree_lambda_min(args, input_size)
    trigger_inv_dir = _trigger_inv_dir(args, args.thres, lambda_min_for_dir)
    if os.path.exists(trigger_inv_dir):
        results['trigger_inv_dir'] = trigger_inv_dir
        
        trigger_files = [f for f in os.listdir(trigger_inv_dir) if f.startswith('fus_')]
        if trigger_files:
            results['trigger_files'] = sorted(trigger_files)
    
    elapsed_time = time.time() - start_time
    results['elapsed_time'] = elapsed_time
    
    
    with open(result_file, 'a') as f:
        f.write(f"{args.weights_path},{l1_norm:.4f},{duration:.4f}\n")
    
    logger.info(f"DECREE detection completed in {elapsed_time:.2f} s")
    if 'l1_norm' in results:
        logger.info(f"L1 norm: {results['l1_norm']:.4f}")

    def _select_test_transform():
        dataset_id = _get_decree_dataset_id(args)
        return transforms.Compose([
            dataset_params[dataset_id]['normalize']
        ])

    def _compute_auprc(scores, labels):
        """Helper function.
            Helper function.
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
        """Helper function.
            Helper function.
            Helper function.
          - tpr, fpr, recall, precision, auroc, auprc
            Helper function.
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
            logger.info(f"{log_prefix}Evaluating clean samples...")
            for x_batch, _ in clean_loader:
                
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

            logger.info(f"{log_prefix}Evaluating poisoned samples...")
            for x_batch, _ in poisoned_loader:
                
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
                f"{log_prefix}Average similarity: clean={metrics['mean_sim_clean']:.6f}, "
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

        
        metrics['threshold'] = float(similarity_threshold)
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['tpr'] = float(tpr)
        metrics['fpr'] = float(fpr)

        return metrics

    def _evaluate_mixed_dataset_detection(mixed_ds, log_prefix, similarity_threshold, poison_keyword="poison"):        """Helper function."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_transform = _select_test_transform()

        loader = DataLoader(mixed_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        all_scores = []
        all_labels = []

        suspicious_model.eval()
        with torch.no_grad():
            logger.info(f"{log_prefix}Evaluating mixed-dataset samples (poison keyword: '{poison_keyword}')...")
            for batch in loader:
                if isinstance(batch, dict):
                    x_batch = batch.get('img')
                    paths = batch.get('img_path')
                else:
                    
                    return {"error": "Mixed dataset must provide rich_output with img_path."}

                if x_batch is None or paths is None:
                    return {"error": "Mixed dataset batch missing img or img_path."}

                
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
                    f"{log_prefix}Average similarity: clean={metrics['mean_sim_clean']:.6f}, "
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

        
        metrics['n_total'] = int(len(all_labels))
        metrics['n_poisoned'] = int(total_positive)
        metrics['n_clean'] = int(total_negative)

        return metrics

    
    can_eval = (
        res_best_trigger and res_best_trigger.get('mask') is not None and res_best_trigger.get('patch') is not None and
        target_backdoor_feature is not None
    )
    if can_eval:
        similarity_threshold = getattr(args, 'similarity_eval_threshold', 0.9)
        logger.info(f"\n====== Starting trigger-based poisoned detection metrics evaluation (threshold={similarity_threshold:.4f}) ======")

        
        if clean_test_dataset is not None and poisoned_test_dataset is not None:
            metrics = _evaluate_poison_detection(
                clean_test_dataset, poisoned_test_dataset,
                log_prefix="[test_dataset] ",
                similarity_threshold=similarity_threshold
            )
            results['sample_classification_metrics'] = metrics

            if "error" not in metrics:
                logger.info(f"[test_dataset] poisoned detection metrics: AUROC={metrics.get('auroc', 0.0):.4f}, AUPRC={metrics.get('auprc', 0.0):.4f}, "
                            f"TPR={metrics.get('tpr', 0.0):.4f}, FPR={metrics.get('fpr', 0.0):.4f}, "
                            f"Precision={metrics.get('precision', 0.0):.4f}, Recall={metrics.get('recall', 0.0):.4f}")
        else:
            logger.info("test clean/poisoned dataset not provided, skip test_dataset metrics.")

        
        if detect_dataset is not None:
            detect_metrics = _evaluate_mixed_dataset_detection(
                detect_dataset,
                log_prefix="[detect_dataset] ",
                similarity_threshold=similarity_threshold,
                poison_keyword="poison"
            )
            results['detect_dataset_metrics'] = detect_metrics
            if "error" not in detect_metrics:
                logger.info(f"[detect_dataset] poisoned detection metrics: AUROC={detect_metrics.get('auroc', 0.0):.4f}, AUPRC={detect_metrics.get('auprc', 0.0):.4f}, "
                            f"TPR={detect_metrics.get('tpr', 0.0):.4f}, FPR={detect_metrics.get('fpr', 0.0):.4f}, "
                            f"Precision={detect_metrics.get('precision', 0.0):.4f}, Recall={detect_metrics.get('recall', 0.0):.4f} "
                            f"(N={detect_metrics.get('n_total', 'N/A')}, P={detect_metrics.get('n_poisoned', 'N/A')}, N0={detect_metrics.get('n_clean', 'N/A')})")
        else:
            logger.info("detect_dataset not provided (single-file mixed list), skip detect_dataset metrics.")
    else:
        if target_backdoor_feature is None:
            logger.info("Target backdoor feature is not available; skip poisoned detection metrics.")
    
    
    logger.removeHandler(file_handler)
    
    return results 
