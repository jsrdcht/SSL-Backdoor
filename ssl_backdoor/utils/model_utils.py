import torch
import torch.nn as nn
import torchvision.models as models
# import ssl_backdoor.ssl_trainers.models_vit as models_vit
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


def get_backbone_model(arch, wts_path, device='cpu', dataset='imagenet100'):
    """获取并加载预训练的主干网络。"""
    if 'moco_' in arch: # histroy
        arch = arch.replace('moco_', '')

    if 'vit' in arch:
        # model = models_vit.__dict__[arch](num_classes=100, global_pool=True)
        # checkpoint_model = load_model_weights(model, wts_path)
        
        # # 移除头部
        # for k in ['head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != model.state_dict()[k].shape:
        #         print(f"移除预训练检查点中的键 {k}")
        #         del checkpoint_model[k]

        # # 插值位置嵌入
        # interpolate_pos_embed(model, checkpoint_model)
        # model.load_state_dict(checkpoint_model, strict=False)
        pass

    else:
        model = models.__dict__[arch]()
        model = remove_task_head_for_encoder(model)
        model = transform_encoder_for_small_dataset(model, dataset)

        state_dict = load_model_weights(model, wts_path)
        print(f"debug for label -1 problem, state_dict: {state_dict.keys()}")
        
        def is_valid_model_param_key(key):
            valid_keys = ['encoder_q', 'backbone', 'encoder', 'model']
            invalid_keys = ['fc', 'head', 'predictor', 'projector', 'projection', 
                          'encoder_k', 'model_t', 'momentum']
                
            # 如果键包含任何无效关键词，过滤掉
            if any([k in key for k in invalid_keys]):
                return False
            
            # 如果键包含任何有效关键词，保留它
            if any([k in key for k in valid_keys]):
                return True
            
            # 如果键既不包含有效关键词也不包含无效关键词，可能是直接的层名
            # 检查是否是常见的网络层命名
            common_layer_patterns = ['conv', 'bn', 'layer', 'downsample', 'running_']
            if any([pattern in key for pattern in common_layer_patterns]):
                return True
            
            return False
        
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
def transform_encoder_for_small_dataset(model: nn.Module, dataset: str):
    assert dataset in ['cifar10', 'cifar100', 'imagenet100', 'imagenet-1k', 'stl10']

    # 判断是不是resnet
    if not 'resnet' in model.__class__.__name__.lower():
        print(f"encoder 不是resnet，不进行适应小数据集的转换")
        return model
    
    if 'cifar10' in dataset or 'cifar100' in dataset:
        model.maxpool = nn.Identity()
    if 'imagenet' not in dataset:
        print(f"transform_encoder_for_small_dataset: {dataset}")
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    return model


def remove_task_head_for_encoder(model: nn.Module):
    if hasattr(model, 'fc'):
        model.fc = nn.Identity()
    elif hasattr(model, 'head'):
        model.head = nn.Identity()
    else:
        raise ValueError(f"model 没有fc或head属性")
    
    return model
