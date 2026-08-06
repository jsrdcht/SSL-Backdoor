"""Hugging Face CLIP/SigLIP encoder adapter."""

from __future__ import annotations

import torch

from ssl_backdoor.utils.model_utils import load_model


def normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class HuggingFaceVisionLanguageEncoder:
    """Expose a common image/text feature interface for CLIP and SigLIP."""

    def __init__(self, model, processor, device: torch.device):
        candidate = model
        if not hasattr(candidate, "get_image_features") and hasattr(candidate, "model"):
            candidate = candidate.model
        required = ("get_image_features", "get_text_features")
        if not all(hasattr(candidate, name) for name in required):
            raise TypeError("model must provide get_image_features and get_text_features")
        if processor is None:
            raise TypeError("CLIP/SigLIP detection requires an image-and-text processor")
        self.model = candidate.to(device).eval()
        self.processor = processor
        self.device = device

    @classmethod
    def from_config(cls, config: dict, device: torch.device):
        model_type = str(config.get("type", "huggingface")).lower()
        if model_type not in {"huggingface", "hf"}:
            raise ValueError("Subspace Detection supports Hugging Face CLIP/SigLIP only")
        model_name = config.get("name")
        if not model_name or not any(key in model_name.lower() for key in ("clip", "siglip")):
            raise ValueError("model.name must identify a Hugging Face CLIP or SigLIP model")
        model, processor = load_model(
            "huggingface",
            model_name,
            config.get("checkpoint"),
            dataset=config.get("dataset", "imagenet"),
            device=str(device),
        )
        return cls(model, processor, device)

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return normalize(self.model.get_image_features(pixel_values=pixel_values.to(self.device)))

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        features = []
        for start in range(0, len(texts), batch_size):
            tokens = self.processor(
                text=texts[start : start + batch_size],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            features.append(normalize(self.model.get_text_features(**tokens)).cpu())
        return torch.cat(features)

    def process_image(self, image) -> torch.Tensor:
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]
