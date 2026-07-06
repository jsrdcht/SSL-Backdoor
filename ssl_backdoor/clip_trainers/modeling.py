"""CLIP model loading and freeze configuration, only supports HuggingFace transformers implementation."""
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class CLIPWrapper(nn.Module):
    """Wraps transformers.CLIPModel, standardizes forward output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def forward(self, pixel_values, input_ids, attention_mask=None):
        out = self.model(pixel_values=pixel_values, input_ids=input_ids,
                         attention_mask=attention_mask, return_dict=True)
        return {
            'image_embeds': out.image_embeds,
            'text_embeds': out.text_embeds,
            'logit_scale': self.model.logit_scale,
        }


def _apply_freeze(model, cfg):
    groups = {
        'train_text_encoder': [model.text_model],
        'train_vision_encoder': [model.vision_model],
        'train_visual_projection': [model.visual_projection],
        'train_text_projection': [model.text_projection],
    }
    for key, modules in groups.items():
        if not cfg.get(key, True):
            for module in modules:
                module.requires_grad_(False)
    model.logit_scale.requires_grad_(bool(cfg.get('train_logit_scale', True)))


def build_clip(model_cfg):
    """Build (CLIPWrapper, CLIPProcessor) according to config."""
    model_type = model_cfg.get('type', 'huggingface')
    if model_type != 'huggingface':
        raise ValueError(f"Only huggingface type CLIP model supported, got: {model_type}")
    name_or_path = model_cfg.get('path') or model_cfg.get('name')
    if not name_or_path:
        raise ValueError("Model config must provide model.path or model.name")

    model = CLIPModel.from_pretrained(name_or_path)
    processor = CLIPProcessor.from_pretrained(name_or_path)
    _apply_freeze(model, model_cfg)
    return CLIPWrapper(model), processor
