from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
import torchvision.models as models

from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ssl_backdoor.constants import HUGGINGFACE_MODEL_PATH


def _strip_prefixes(key: str, prefixes: Tuple[str, ...]) -> str:
    """仅在前缀位置 strip，避免误删中间子串。"""
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def _transform_encoder_for_small_dataset(model: nn.Module) -> nn.Module:
    """
    针对小分辨率数据集（如 CIFAR）的 ResNet 常见结构改造：
    - 去掉 maxpool
    - 将首层 conv1 改为 3x3, stride=1

    注意：如果模型不具备对应属性（非 ResNet 或结构不同），则保持原样返回。
    """
    if not hasattr(model, "maxpool") or not hasattr(model, "conv1"):
        return model

    model.maxpool = nn.Identity()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    return model

def transform_encoder_for_small_dataset(model: nn.Module, dataset: Optional[str] = None) -> nn.Module:
    """
    兼容旧代码的对外接口：根据 dataset 判断是否需要对 encoder 进行小数据集适配。
    """
    if dataset is None:
        return model
    from ssl_backdoor.constants import DATASET_THAT_NEED_TO_TRANSFORM_ENCODER
    if dataset in DATASET_THAT_NEED_TO_TRANSFORM_ENCODER:
        _transform_encoder_for_small_dataset(model)
    return model



def remove_task_head_for_encoder(model: nn.Module):
    for attr in ('fc', 'head', 'classifier', 'heads'):
        if hasattr(model, attr):
            setattr(model, attr, nn.Identity())
            return model
    raise ValueError(f"model 没有 fc、head、classifier 或 heads 属性")

    
# load model weights to cpu
def load_checkpoint(wts_path: str) -> Dict[str, Any]:
    """加载并处理模型权重文件。"""
    checkpoint = torch.load(wts_path, map_location='cpu')
    for key in ['model', 'state_dict', 'model_state_dict']:
        if key in checkpoint:
            return checkpoint[key]
    raise ValueError(f'No model or state_dict found in {wts_path}.')


def get_backbone_model(arch, wts_path, device='cpu', dataset='imagenet100', freeze_backbone=False):
    """获取并加载预训练的主干网络。"""

    model = models.__dict__[arch]()
    model = remove_task_head_for_encoder(model)
    from ssl_backdoor.constants import DATASET_THAT_NEED_TO_TRANSFORM_ENCODER
    if dataset in DATASET_THAT_NEED_TO_TRANSFORM_ENCODER and 'resnet' in arch.lower():
        _transform_encoder_for_small_dataset(model)

    if wts_path is None:
        warnings.warn("wts_path is None, return init model", UserWarning)
        return model
        

    state_dict = load_checkpoint(wts_path)
    print(f"state_dict keys: {state_dict.keys()}")
    def is_valid_model_param_key(key):
        key = _strip_prefixes(key, ('module.', 'model.'))
        valid_keys = ['encoder_q', 'backbone', 'encoder', 'model']
        invalid_keys = ['fc', 'head', 'predictor', 'projector', 'projection', 
                        'encoder_k', 'model_t', 'momentum', 'regressor']
            
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
    
    
    state_dict = {_strip_prefixes(k, ('module.', 'model.', 'encoder_q.', 'encoder.', 'backbone.')): v for k, v in state_dict.items() if is_valid_model_param_key(k)}

    incompatible = model.load_state_dict(state_dict, strict=False)
    if getattr(incompatible, "unexpected_keys", None):
        print(f"[get_backbone_model] unexpected_keys({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys}")
    if getattr(incompatible, "missing_keys", None):
        raise RuntimeError(f"[get_backbone_model] missing_keys({len(incompatible.missing_keys)}): {incompatible.missing_keys}")
        
    model = model.to(device)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    
    return model


def load_huggingface_representation_model(model_name, device='cpu'):
    """
    根据模型名称加载对应的参考模型和处理器
    
    Args:
        model_name (str): 模型名称，例如'clip'或'dinov2-base'
        device (str or torch.device): 模型运行的设备
        
    Returns:
        tuple: (model, processor)
    """

    if 'clip' in model_name.lower():
        model = CLIPModel.from_pretrained(f"{HUGGINGFACE_MODEL_PATH}/{model_name}")
        processor = CLIPProcessor.from_pretrained(f"{HUGGINGFACE_MODEL_PATH}/{model_name}")
    elif 'siglip' in model_name.lower():
        model = AutoModel.from_pretrained(f"{HUGGINGFACE_MODEL_PATH}/{model_name}/model")
        processor = AutoProcessor.from_pretrained(f"{HUGGINGFACE_MODEL_PATH}/{model_name}/processor")
    else:
        # 对于其他huggingface模型，尝试自动加载
        model = AutoModel.from_pretrained(f"{HUGGINGFACE_MODEL_PATH}/{model_name}")
        processor = AutoImageProcessor.from_pretrained(f"{HUGGINGFACE_MODEL_PATH}/{model_name}")
    
    model.eval()
    model = model.to(device)
    return model, processor


def load_model(model_type: str, model_name: str, model_path: str, dataset: str = 'cifar10', device: str = 'cpu') -> Tuple[nn.Module, Optional[Any]]:
    """
    加载模型（支持 pytorch 与 huggingface）。

    Args:
        model_type (str): 模型来源类型，取值为 'pytorch' 或 'huggingface'（兼容常见拼写变体）。
        model_name (str): 模型名称。
        model_path (str): 训练好的模型权重路径。
        dataset (str): 数据集名称，用于 pytorch 模型的结构适配。
        device (str): 模型加载到的设备。

    Returns:
        tuple[nn.Module, Any | None]: (模型, 处理器)。pytorch 返回 (model, None)，huggingface 返回 (model, processor)。
    """
    processor: Optional[Any] = None

    normalized_type = model_type.strip().lower()
    huggingface_alias = {'huggingface', 'hf', 'hugginface', 'hugggingface'}
    pytorch_alias = {'pytorch', 'torch'}

    if normalized_type in huggingface_alias:
        # 1) 先从本地 MODEL_PATH 下加载同名模型与处理器
        model, processor = load_huggingface_representation_model(model_name, device=device)

        # 2) 再加载训练好的权重
        state_dict = load_checkpoint(model_path)
        state_dict = {_strip_prefixes(k, ('module.', 'model.')): v for k, v in state_dict.items()}

        # 3) 兼容不同工程保存的 CLIP 命名，将其映射到 HuggingFace 的键名
        def _map_clip_keys_to_hf(sd: Dict[str, Any]) -> Dict[str, Any]:
            mapped = {}
            for k, v in sd.items():
                new_k = k
                # vision branch
                new_k = new_k.replace('vision_embeddings', 'vision_model.embeddings')
                new_k = new_k.replace('vision_encoder', 'vision_model.encoder')
                new_k = new_k.replace('vision_pre_layernorm', 'vision_model.pre_layernorm')
                new_k = new_k.replace('vision_post_layernorm', 'vision_model.post_layernorm')
                # text branch
                new_k = new_k.replace('text_embeddings', 'text_model.embeddings')
                new_k = new_k.replace('text_encoder', 'text_model.encoder')
                new_k = new_k.replace('text_final_layer_norm', 'text_model.final_layer_norm')
                mapped[new_k] = v
            return mapped

        state_dict = _map_clip_keys_to_hf(state_dict)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if getattr(incompatible, "unexpected_keys", None):
            print(f"[load_model] unexpected_keys({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys}")
        if getattr(incompatible, "missing_keys", None):
            raise RuntimeError(f"[load_model] missing_keys({len(incompatible.missing_keys)}): {incompatible.missing_keys}")
    elif normalized_type in pytorch_alias:
        # 直接通过 backbone 加载（内部已处理是否去除任务头、微调小数据集等）
        model = get_backbone_model(model_name, model_path, device=device, dataset=dataset, freeze_backbone=False)
    else:
        raise ValueError(f"未知的 model_type: {model_type}. 期望 'pytorch' 或 'huggingface'")

    model.eval()
    model = model.to(device)
    return model, processor


def get_features(model, dataloader, device, processor=None, normalize=False):
    """从模型中提取特征"""
    features = []
    paths = []
    labels = []
    
    model.eval()
    with torch.inference_mode():
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


