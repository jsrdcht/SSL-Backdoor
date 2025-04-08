import torch
import torch.nn as nn
import torchvision.models as models
import ssl_backdoor.ssl_trainers.models_vit as models_vit
from typing import Dict, Any
from ssl_backdoor.utils.utils import interpolate_pos_embed, get_channels


def load_model_weights(model, wts_path: str) -> Dict[str, Any]:
    """
    加载模型权重
    
    Args:
        model: 待加载权重的模型
        wts_path: 权重文件路径
        
    Returns:
        模型状态字典
        
    Raises:
        ValueError: 如果找不到有效的权重
    """
    checkpoint = torch.load(wts_path, map_location='cpu')
    
    # 按优先级顺序尝试不同的键名
    for key in ['model', 'state_dict', 'model_state_dict']:
        if key in checkpoint:
            print(f"根据 {key} 作为键从 {wts_path} 加载模型权重")
            return checkpoint[key]
            
    raise ValueError(f'无法在 {wts_path} 中找到模型权重')


def get_backbone_model(arch, wts_path, device='cpu'):
    """获取并加载预训练的主干网络。"""
    if 'moco_' in arch: # histroy
        arch = arch.replace('moco_', '')

    if 'vit' in arch:
        model = models_vit.__dict__[arch](num_classes=100, global_pool=True)
        checkpoint_model = load_model_weights(model, wts_path)
        
        # 移除头部
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != model.state_dict()[k].shape:
                print(f"移除预训练检查点中的键 {k}")
                del checkpoint_model[k]

        # 插值位置嵌入
        interpolate_pos_embed(model, checkpoint_model)
        model.load_state_dict(checkpoint_model, strict=False)

    else:
        model = models.__dict__[arch]()
        
        if hasattr(model, 'fc'):  model.fc = nn.Sequential()
        if hasattr(model, 'head'):  model.head = nn.Sequential()

        state_dict = load_model_weights(model, wts_path)
        
        def is_valid_model_param_key(key):
            valid_keys = ['encoder_q', 'backbone', 'encoder', 'model']
            invalid_keys = ['fc', 'head', 'predictor', 'projector', 'projection', 
                          'encoder_k', 'model_t', 'momentum']
                
            if any([k in key for k in invalid_keys]):
                return False
            if not any([k in key for k in valid_keys]):
                return False
            return True
        
        def model_param_key_filter(key):
            if 'model.' in key:
                key = key.replace('model.', '')
            if 'module.' in key:
                key = key.replace('module.', '')
            if 'encoder.' in key:
                key = key.replace('encoder.', '')
            if 'encoder_q.' in key:
                key = key.replace('encoder_q.', '')
            if 'backbone.' in key:
                key = key.replace('backbone.', '')
            return key
           
        state_dict = {model_param_key_filter(k): v for k, v in state_dict.items() if is_valid_model_param_key(k)}
        model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    
    return model