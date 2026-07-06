import numpy as np
import torch
from PIL import Image
import os

def epsilon():
    """Helper function.."""
    return 1e-6

def assert_range(tensor, min_val, max_val):
    """Helper function.."""
    assert torch.min(tensor) >= min_val and torch.max(tensor) <= max_val,\
        f"Value out of range [{min_val}, {max_val}], actual range: [{torch.min(tensor).item()}, {torch.max(tensor).item()}]"

def compute_self_cos_sim(feat):
    """Helper function.."""
    feat_norm = torch.norm(feat, dim=1, keepdim=True)
    normalized_feat = feat / feat_norm
    sim_matrix = torch.mm(normalized_feat, normalized_feat.t())
    
    
    mask = torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
    non_diag_sim = sim_matrix.masked_select(~mask).reshape(sim_matrix.shape[0], -1)
    return torch.mean(non_diag_sim)

def dump_img(tensor, path):
    """Helper function.."""
    if tensor.ndim == 4:  
        for i in range(tensor.shape[0]):
            img_tensor = tensor[i]
            dump_img(img_tensor, f"{path}_{i}.png")
        return
    
    if tensor.shape[0] == 3:  
        tensor = tensor.permute(1, 2, 0)
    
    
    if tensor.max() <= 1.0:
        tensor = tensor * 255.0
    
    tensor = tensor.detach().cpu().numpy().astype(np.uint8)
    img = Image.fromarray(tensor)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def generate_mask(mask_size, t_x, t_y, r):
    """Helper function.."""
    mask = np.zeros([mask_size, mask_size]) + epsilon()
    patch = np.random.rand(mask_size, mask_size, 3)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (t_x <= i and i < t_x + r) and\
               (t_y <= j and j < t_y + r): 
                mask[i][j] = 1.0
    return mask, patch

from ssl_backdoor.utils.utils import set_seed
