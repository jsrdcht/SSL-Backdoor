from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
# import ssl_backdoor.ssl_trainers.models_vit as models_vit
from typing import Dict, Any
from ssl_backdoor.utils.utils import interpolate_pos_embed, get_channels

from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel, AutoProcessor


def load_checkpoint(wts_path: str) -> Dict[str, Any]:
    """加载并处理模型权重文件。"""
    checkpoint = torch.load(wts_path, map_location='cpu')
    if 'model' in checkpoint:
        return checkpoint['model']
    elif 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    else:
        raise ValueError(f'No model or state_dict found in {wts_path}.')


def get_backbone_model(arch, wts_path, device='cpu', dataset='imagenet100', freeze_backbone=True):
    """获取并加载预训练的主干网络。"""
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
        
        if arch != 'SqueezeNet':
            model = models.__dict__[arch]()
            model = remove_task_head_for_encoder(model)
            model = transform_encoder_for_small_dataset(model, dataset)
        else:
            model = models.__dict__[arch](num_classes=2048)

        if wts_path is None:
            print(f"wts_path is None, return init model")
            return model
        
        state_dict = load_checkpoint(wts_path)
        print(f"state_dict keys: {state_dict.keys()}")
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

        msg = model.load_state_dict(state_dict, strict=True)
        print(f"msg: {msg}")
        
    model = model.to(device)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    
    return model


def load_huggingface_model(model_name, model_path):
    """
    根据模型名称加载对应的参考模型和处理器
    
    Args:
        model_name (str): 模型名称，例如'clip'或'dinov2-base'
        device (torch.device): 模型运行的设备
        
    Returns:
        tuple: (model, processor)
    """

    MODEL_PATH = "/workspace/pretrained_models"
    if model_name.lower() == 'clip':
        model = CLIPModel.from_pretrained(f"{MODEL_PATH}/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained(f"{MODEL_PATH}/clip-vit-base-patch32")
    elif model_name.lower() == 'siglip':
        model = AutoModel.from_pretrained(f"{MODEL_PATH}/siglip-base-patch16-224/model")
        processor = AutoProcessor.from_pretrained(f"{MODEL_PATH}/siglip-base-patch16-224/processor")
    else:
        # 对于其他huggingface模型，尝试自动加载
        model = AutoModel.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
    
    model.eval()
    return model, processor


def load_model(model_type: str, model_path: str, dataset='cifar10') -> nn.Module:
    """
    加载后门模型
    
    Args:
        model_type (str): 模型类型，如 'resnet18'
        model_path (str): 模型权重路径
        dataset (str): 数据集名称
        
    Returns:
        nn.Module: 加载的模型
    """
    processor = None
    model_type = model_type.lower()
    DEVICE = 'cpu'
    HUGGINFACE_MODEL_LIST = ['clip', 'siglip']
    
    if any([model in model_type for model in HUGGINFACE_MODEL_LIST]):
        model, processor = load_huggingface_model(model_type, model_path)

        state_dict = load_checkpoint(model_path)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=True)
        print(f"load model msg: {msg}")
    else:
        print(f"加载标准模型: {model_type}")
        # has been loaded in get_backbone_model
        model = get_backbone_model(model_type, model_path, device=DEVICE, dataset=dataset, freeze_backbone=True)
    
    model.eval()
    return model, processor


def get_features(model, dataloader, device, processor=None, normalize=True):
    """从模型中提取特征"""
    features = []
    paths = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if processor is not None:  # 使用处理器的模型（CLIP, DINOv2等）
                images, targets, img_paths = batch
                # 使用处理器进行预处理
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            else:  # ResNet model
                images, targets, img_paths = batch
                images = images.to(device)
                inputs = images
            
            # 根据不同模型类型处理输出
            if hasattr(model, 'get_image_features'):  # CLIP model
                if isinstance(inputs, dict):
                    image_features = model.get_image_features(**inputs)
                else:
                    image_features = model.get_image_features(inputs)

                if normalize:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            else:  # 其他模型
                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                else:
                    outputs = model(inputs)
                
                # 处理不同类型的输出
                if isinstance(outputs, BaseModelOutputWithPooling):  # DINOv2等Transformers模型
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        image_features = outputs.pooler_output
                    # 如果没有pooler_output，使用last_hidden_state的第一个token（CLS token）
                    elif hasattr(outputs, 'last_hidden_state'):
                        image_features = outputs.last_hidden_state[:, 0]
                    else:
                        raise ValueError("No valid feature extraction method found.")
                else:  # 直接返回tensor的模型
                    image_features = outputs
                
                # 对于所有特征，进行L2归一化
                if isinstance(image_features, torch.Tensor) and normalize:
                    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
            
            features.append(image_features.cpu())
            paths.extend(img_paths)
            labels.extend(targets.numpy())
    
    features = torch.cat(features, dim=0)
    return features, paths, labels 


def transform_encoder_for_small_dataset(model: nn.Module, dataset: str):
    assert dataset in ['cifar10', 'cifar100', 'imagenet100', 'imagenet-1k', 'imagenet', 'stl10', 'gtsrb']

    # 判断是不是resnet
    if not 'resnet' in model.__class__.__name__.lower():
        print(f"encoder 不是resnet，不进行适应小数据集的转换")
        return model
    
    if 'cifar10' in dataset or 'cifar100' in dataset or 'gtsrb' in dataset:
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
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Identity()
    elif hasattr(model, 'heads'):
        model.heads = nn.Identity()
    else:
        raise ValueError(f"model 没有fc或head或classifier属性")
    
    return model
