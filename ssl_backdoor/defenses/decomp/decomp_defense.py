
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from .hf_prs_hook import hook_prs_logger

def extract_prs_features(model, dataloader, device):
    """
    Extracts PRS features for a dataset and computes the mean (prototype) vectors.
    """
    model.eval()
    prs = hook_prs_logger(model, device)
    
    attentions_sum = None
    mlps_sum = None
    count = 0
    
    print("Extracting PRS prototypes...")
    try:
        for batch in tqdm(dataloader):
            # batch is (images, labels) or just images
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
                
            images = images.to(device)
            
            with torch.no_grad():
                prs.reinit()
                # We need to trigger the hooks.
                # model.get_image_features calls vision_model
                # Note: prs.finalize needs the final representation
                image_features = model.get_image_features(images) 
                
                # Finalize to get [B, L, H, D] and [B, L+1, D]
                attentions, mlps = prs.finalize(image_features)
                
                # attentions: [B, Layers, Heads, Output_Dim]
                # mlps: [B, Layers+1, Output_Dim]
                
                if attentions_sum is None:
                    attentions_sum = torch.zeros_like(attentions[0]) # [L, H, D]
                    mlps_sum = torch.zeros_like(mlps[0])             # [L+1, D]
                
                attentions_sum += attentions.sum(dim=0)
                mlps_sum += mlps.sum(dim=0)
                count += images.size(0)
                
        attentions_mean = attentions_sum / count
        mlps_mean = mlps_sum / count
    
    finally:
        prs.close()
    
    return attentions_mean, mlps_mean

def get_zero_shot_classifier(model, class_names, device, processor=None, templates=None):
    if templates is None:
        raise ValueError("templates must be provided")
    
    model.eval()
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(class_names, desc="Building Zero-Shot Classifier"):
            texts = [template.format(classname) for template in templates]
            
            if processor is not None:
                # HF style
                # Fix: Use padding="max_length" to ensure consistent EOS position behavior with original CLIP
                inputs = processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(device)
                class_embeddings = model.get_text_features(**inputs)
            else:
                # Assuming model has a tokenizer method (e.g. OpenCLIP or custom wrapper)
                # But here we focus on HF support as per plan
                # Fallback or error?
                # Let's assume user must pass processor for HF models.
                raise ValueError("processor must be provided for HF models")
                
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def run_image_detection_and_ablation(model, dataloader, attns_mean, mlps_mean, device, 
                                     ablation_target='head', mlp_target_layers=5,
                                     head_threshold=0.002, detection_threshold=5, 
                                     ablation_type='mean', target_label=None, 
                                     classifier_weights=None):
    """
    Runs detection and ablation on a dataset.
    
    Args:
        attns_mean: [Layers, Heads, Output_Dim] (Prototypes)
        mlps_mean: [Layers+1, Output_Dim]
    """
    model.eval()
    prs = hook_prs_logger(model, device)
    
    # Move means to device
    attns_mean = attns_mean.to(device)
    mlps_mean = mlps_mean.to(device)
    
    all_preds_clean = []
    all_preds_ablated = []
    all_labels = []
    is_poisoned_pred = []
    num_bad_heads_list = []
    num_bad_mlps_list = []
    
    print(f"Running Detection and Ablation (Target: {ablation_target})...")
    try:
        for batch in tqdm(dataloader):
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            else:
                images = batch
                labels = None # Should handle if labels are missing
                
            images = images.to(device)
            if labels is not None:
                labels = labels.to(device)
                
            with torch.no_grad():
                prs.reinit()
                image_features = model.get_image_features(images)
                attentions, mlps = prs.finalize(image_features) # [B, L, H, D], [B, L+1, D]
                
                # --- Detection Logic (match original head-based logic) ---
                # Original implementation for BadNet-style attacks detects/edits based on LAST layer heads only.
                last_layer_attns = attentions[:, -1, :, :]  # [B, H, D]
                last_layer_proto = attns_mean[-1, :, :]     # [H, D]
                head_sims = (last_layer_attns * last_layer_proto.unsqueeze(0)).sum(dim=-1)  # [B, H]
                bad_heads_mask = head_sims < head_threshold  # [B, H]
                bad_head_counts = bad_heads_mask.sum(dim=1)  # [B]
                is_bd_pred = (bad_head_counts > detection_threshold).long()  # [B]

                # --- Build ablation masks ---
                # full_attn_mask: [B, L, H], True means replace
                full_attn_mask = torch.zeros(attentions.shape[:3], dtype=torch.bool, device=device)
                # bad_mlps_mask: [B, L+1], True means replace
                bad_mlps_mask = torch.zeros(mlps.shape[:2], dtype=torch.bool, device=device)

                if ablation_target == 'head':
                    # Mean-ablate ONLY the last layer heads flagged by threshold
                    full_attn_mask[:, -1, :] = bad_heads_mask
                elif ablation_target == 'mlp':
                    # Mean-ablate the LAST N MLP components (match original mlp_editing: fixed last-5 layers)
                    num_mlp_layers = mlps.shape[1]
                    n = min(int(mlp_target_layers), num_mlp_layers)
                    if n > 0:
                        bad_mlps_mask[:, -n:] = True
                else:
                    raise ValueError(f"Unknown ablation_target: {ablation_target}")

                bad_mlp_counts = bad_mlps_mask.sum(dim=1)

                # Store counts
                num_bad_heads_list.append(bad_head_counts.cpu())
                num_bad_mlps_list.append(bad_mlp_counts.cpu())
                is_poisoned_pred.append(is_bd_pred.cpu())
                
                # --- Ablation Logic ---
                
                if ablation_type == 'mean':
                    attn_replacements = attns_mean.unsqueeze(0).expand_as(attentions)
                    mlp_replacements = mlps_mean.unsqueeze(0).expand_as(mlps)
                elif ablation_type == 'zero':
                    attn_replacements = torch.zeros_like(attentions)
                    mlp_replacements = torch.zeros_like(mlps)
                else:
                    raise ValueError(f"Unknown ablation type: {ablation_type}")
                    
                # Apply ablation
                ablated_attentions = torch.where(full_attn_mask.unsqueeze(-1), attn_replacements, attentions)
                ablated_mlps = torch.where(bad_mlps_mask.unsqueeze(-1), mlp_replacements, mlps)
                
                # Reconstruct feature vector
                ablated_attn_sum = ablated_attentions.sum(dim=(1, 2))
                ablated_mlp_sum = ablated_mlps.sum(dim=1) 
                
                ablated_features = ablated_attn_sum + ablated_mlp_sum
                
                # Original features (for comparison/baseline)
                # original_features = attentions.sum(dim=(1,2)) + mlps.sum(dim=1) 
                
                # --- Classification ---
                if classifier_weights is not None:
                    # CLIP uses cosine similarity (unit-norm features).
                    image_features_norm = F.normalize(image_features, dim=-1)
                    ablated_features_norm = F.normalize(ablated_features, dim=-1)
                    
                    logits_clean = image_features_norm @ classifier_weights
                    logits_ablated = ablated_features_norm @ classifier_weights
                    
                    preds_clean = logits_clean.argmax(dim=1)
                    preds_ablated = logits_ablated.argmax(dim=1)
                    
                    all_preds_clean.append(preds_clean.cpu())
                    all_preds_ablated.append(preds_ablated.cpu())
                    if labels is not None:
                        all_labels.append(labels.cpu())
    
        # Aggregate results
        num_bad_heads_list = torch.cat(num_bad_heads_list)
        num_bad_mlps_list = torch.cat(num_bad_mlps_list)
        is_poisoned_pred = torch.cat(is_poisoned_pred)
        
        metrics = {
            "avg_bad_heads": num_bad_heads_list.float().mean().item(),
            "avg_bad_mlps": num_bad_mlps_list.float().mean().item(),
            "detection_rate": is_poisoned_pred.float().mean().item()
        }
        
        if classifier_weights is not None:
            all_preds_clean = torch.cat(all_preds_clean)
            all_preds_ablated = torch.cat(all_preds_ablated)
            
            if len(all_labels) > 0:
                all_labels = torch.cat(all_labels)
                
                acc_clean = (all_preds_clean == all_labels).float().mean().item()
                acc_ablated = (all_preds_ablated == all_labels).float().mean().item()
                
                metrics["acc_clean"] = acc_clean
                metrics["acc_ablated"] = acc_ablated
            
            if target_label is not None:
                asr_clean = (all_preds_clean == target_label).float().mean().item()
                asr_ablated = (all_preds_ablated == target_label).float().mean().item()
                metrics["asr_clean"] = asr_clean
                metrics["asr_ablated"] = asr_ablated

    finally:
        prs.close()

    return metrics, num_bad_heads_list, is_poisoned_pred

def calculate_detection_metrics(clean_scores, poison_scores, clean_preds, poison_preds):
    """
    Calculates TPR, FPR, AUROC, AUPRC given scores and predictions from clean and poisoned data.
    
    Args:
        clean_scores (torch.Tensor): Anomaly scores for clean data (e.g. num_bad_heads). Higher means more likely poisoned.
        poison_scores (torch.Tensor): Anomaly scores for poisoned data.
        clean_preds (torch.Tensor): Binary predictions for clean data (1=poisoned, 0=clean).
        poison_preds (torch.Tensor): Binary predictions for poisoned data.
        
    Returns:
        dict: Dictionary containing TPR, FPR, AUROC, AUPRC.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # TPR and FPR
    # TPR = TP / (TP + FN) = proportion of poisoned samples correctly identified as poisoned
    tpr = poison_preds.float().mean().item()
    
    # FPR = FP / (FP + TN) = proportion of clean samples incorrectly identified as poisoned
    fpr = clean_preds.float().mean().item()
    
    # Prepare data for AUROC / AUPRC
    # cleaning up scores: ensure they are cpu numpy or tensor
    if isinstance(clean_scores, torch.Tensor):
        clean_scores = clean_scores.cpu().detach()
    if isinstance(poison_scores, torch.Tensor):
        poison_scores = poison_scores.cpu().detach()
        
    y_true = torch.cat([torch.zeros(len(clean_scores)), torch.ones(len(poison_scores))])
    y_score = torch.cat([clean_scores, poison_scores])
    
    try:
        auroc = roc_auc_score(y_true.numpy(), y_score.numpy())
    except ValueError:
        auroc = 0.0
        
    try:
        auprc = average_precision_score(y_true.numpy(), y_score.numpy())
    except ValueError:
        auprc = 0.0
        
    return {
        "TPR": tpr,
        "FPR": fpr,
        "AUROC": auroc,
        "AUPRC": auprc
    }
