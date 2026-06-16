import numpy as np
import torch
from PIL import Image
import os

def epsilon():
    """返回一个非常小的正数，用于数值稳定性"""
    return 1e-6

def assert_range(tensor, min_val, max_val):
    """断言张量值在指定范围内"""
    assert torch.min(tensor) >= min_val and torch.max(tensor) <= max_val, \
        f"值超出范围 [{min_val}, {max_val}]，实际范围: [{torch.min(tensor).item()}, {torch.max(tensor).item()}]"

def compute_self_cos_sim(feat):
    """计算特征的余弦相似度"""
    feat_norm = torch.norm(feat, dim=1, keepdim=True)
    normalized_feat = feat / feat_norm
    sim_matrix = torch.mm(normalized_feat, normalized_feat.t())
    
    # 计算平均自相似度（对角线上的元素应为1）
    mask = torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
    non_diag_sim = sim_matrix.masked_select(~mask).reshape(sim_matrix.shape[0], -1)
    return torch.mean(non_diag_sim)

def dump_img(tensor, path):
    """将张量保存为图像"""
    if tensor.ndim == 4:  # 批处理图像
        for i in range(tensor.shape[0]):
            img_tensor = tensor[i]
            dump_img(img_tensor, f"{path}_{i}.png")
        return
    
    if tensor.shape[0] == 3:  # CHW 格式
        tensor = tensor.permute(1, 2, 0)
    
    # 确保张量值在 [0, 255] 范围内
    if tensor.max() <= 1.0:
        tensor = tensor * 255.0
    
    tensor = tensor.detach().cpu().numpy().astype(np.uint8)
    img = Image.fromarray(tensor)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def generate_mask(mask_size, t_x, t_y, r):
    """生成触发器掩码和补丁"""
    mask = np.zeros([mask_size, mask_size]) + epsilon()
    patch = np.random.rand(mask_size, mask_size, 3)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (t_x <= i and i < t_x + r) and \
               (t_y <= j and j < t_y + r): 
                mask[i][j] = 1.0
    return mask, patch

from ssl_backdoor.utils.utils import set_seed
 