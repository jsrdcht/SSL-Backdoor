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
    """."""
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
        
        
        

        
    """
    if not hasattr(model, "maxpool") or not hasattr(model, "conv1"):
        return model

    model.maxpool = nn.Identity()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    return model

def transform_encoder_for_small_dataset(model: nn.Module, dataset: Optional[str] = None) -> nn.Module:
    """
        
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
    raise ValueError(f"the model has no fc, head, classifier, or heads attribute")

    
# load model weights to cpu
def load_checkpoint(wts_path: str) -> Dict[str, Any]:
    """."""
    checkpoint = torch.load(wts_path, map_location='cpu')
    # 1) {"state_dict": OrderedDict(...)} / {"model": ...} / {"model_state_dict": ...}
    if isinstance(checkpoint, dict):
        for key in ['model', 'state_dict', 'model_state_dict']:
            if key in checkpoint:
                return checkpoint[key]
        if all(isinstance(k, str) for k in checkpoint.keys()) and any(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    raise ValueError(f'No model or state_dict found in {wts_path}.')


def get_backbone_model(arch, wts_path, device='cpu', dataset='imagenet100', freeze_backbone=False):
    """."""

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
        if any([k in key for k in invalid_keys]):
            return False
        if any([k in key for k in valid_keys]):
            return True
        common_layer_patterns = ['conv', 'bn', 'layer', 'downsample', 'running_']
        if any([pattern in key for pattern in common_layer_patterns]):
            return True
        
        return False
    
    
    state_dict = {_strip_prefixes(k, ('module.', 'model.', 'encoder_q.', 'base_encoder.', 'encoder.', 'backbone.')): v for k, v in state_dict.items() if is_valid_model_param_key(k)}

    # Legacy MoCo STL10 encoders used a 3x3 conv1 but kept the standard maxpool.
    if dataset == 'stl10' and 'resnet' in arch.lower():
        conv1_weight = state_dict.get('conv1.weight')
        if conv1_weight is not None and conv1_weight.shape[2:] == (3, 3):
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

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
        
    
    Args:
        
        
        
    Returns:
        tuple: (model, processor)
    """

    if 'clip' in model_name.lower():
        model_dir = f"{HUGGINGFACE_MODEL_PATH}/{model_name}"
        model = CLIPModel.from_pretrained(model_dir)
        try:
            processor = CLIPProcessor.from_pretrained(model_dir)
        except Exception as e:
            msg = str(e)
            if "ModelWrapper" in msg or "TokenizerFast.from_file" in msg:
                print(f"[load_huggingface_representation_model] CLIP fast tokenizer Loading failed, fallback to use_fast=False: {e}")
                processor = CLIPProcessor.from_pretrained(model_dir, use_fast=False)
            else:
                raise
    elif 'siglip' in model_name.lower():
        model = AutoModel.from_pretrained(f"{HUGGINGFACE_MODEL_PATH}/{model_name}/model")
        processor = AutoProcessor.from_pretrained(f"{HUGGINGFACE_MODEL_PATH}/{model_name}/processor")
    else:
        model = AutoModel.from_pretrained(f"{HUGGINGFACE_MODEL_PATH}/{model_name}")
        processor = AutoImageProcessor.from_pretrained(f"{HUGGINGFACE_MODEL_PATH}/{model_name}")
    
    model.eval()
    model = model.to(device)
    return model, processor


def load_model(model_type: str, model_name: str, model_path: Optional[str], dataset: str = 'cifar10', device: str = 'cpu') -> Tuple[nn.Module, Optional[Any]]:
    """
        

    Args:
        
        
        
        
        

    Returns:
        
    """
    processor: Optional[Any] = None

    normalized_type = model_type.strip().lower()
    huggingface_alias = {'huggingface', 'hf', 'hugginface', 'hugggingface'}
    pytorch_alias = {'pytorch', 'torch'}

    if normalized_type in huggingface_alias:
        model, processor = load_huggingface_representation_model(model_name, device=device)
        if model_path is not None:
            state_dict = load_checkpoint(model_path)
            state_dict = {_strip_prefixes(k, ('module.', 'model.')): v for k, v in state_dict.items()}
            def _map_clip_keys_to_hf(sd: Dict[str, Any]) -> Dict[str, Any]:
                mapped = {}
                for k, v in sd.items():
                    new_k = k
                    # vision branch
                    new_k = new_k.replace('vision_embeddings', 'vision_model.embeddings')
                    new_k = new_k.replace('vision_encoder', 'vision_model.encoder')
                    new_k = new_k.replace('vision_pre_layernorm', 'vision_model.pre_layrnorm')
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
        model = get_backbone_model(model_name, model_path, device=device, dataset=dataset, freeze_backbone=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'pytorch' or 'huggingface'")

    model.eval()
    model = model.to(device)
    return model, processor


def get_features(model, dataloader, device, processor=None, normalize=False):
    """."""
    features = []
    paths = []
    labels = []
    
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if processor is not None:
                images, targets, img_paths = batch
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            else:  # ResNet model
                images, targets, img_paths = batch
                images = images.to(device)
                inputs = images
            if hasattr(model, 'get_image_features'):  # CLIP model
                if isinstance(inputs, dict):
                    image_features = model.get_image_features(**inputs)
                else:
                    image_features = model.get_image_features(inputs)

                if normalize:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                else:
                    outputs = model(inputs)
                if isinstance(outputs, BaseModelOutputWithPooling):
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        image_features = outputs.pooler_output
                    elif hasattr(outputs, 'last_hidden_state'):
                        image_features = outputs.last_hidden_state[:, 0]
                    else:
                        raise ValueError("No valid feature extraction method found.")
                else:
                    image_features = outputs
                if isinstance(image_features, torch.Tensor) and normalize:
                    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
            
            features.append(image_features.cpu())
            paths.extend(img_paths)
            labels.extend(targets.numpy())
    
    features = torch.cat(features, dim=0)
    return features, paths, labels 

