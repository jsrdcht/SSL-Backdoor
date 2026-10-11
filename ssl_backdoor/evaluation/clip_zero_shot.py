"""Algorithm-independent CLIP zero-shot classification."""

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PromptTemplate = Union[str, Callable[[str], str]]


def _format_prompt(template: PromptTemplate, class_name: str) -> str:
    return template(class_name) if callable(template) else template.format(class_name)


def _encode_text(model, tokenizer, prompts, device, max_text_length):
    # Each model API is paired with its tokenizer's input and output conventions.
    if hasattr(model, "get_text_features"):
        kwargs = {
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
        }
        if max_text_length is not None:
            kwargs["max_length"] = max_text_length
        tokenized = tokenizer(text=prompts, **kwargs)
        inputs = {key: value.to(device) for key, value in tokenized.items()}
        return model.get_text_features(**inputs)
    if hasattr(model, "encode_text"):
        tokens = tokenizer(prompts).to(device)
        return model.encode_text(tokens)
    raise TypeError("CLIP models must provide get_text_features or encode_text")


def _encode_image(model, pixel_values):
    if hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values=pixel_values)
    if hasattr(model, "encode_image"):
        return model.encode_image(pixel_values)
    raise TypeError("CLIP models must provide get_image_features or encode_image")


def _require_feature_matrix(features, source):
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError(f"{source} must return a tensor with shape [B, D]")
    return features


@torch.inference_mode()
def build_clip_text_prototypes(
    model: nn.Module,
    tokenizer,
    class_names: Sequence[str],
    templates: Sequence[PromptTemplate],
    device: Union[str, torch.device],
    *,
    max_text_length: Optional[int] = 77,
) -> torch.Tensor:
    """Build normalized text prototypes with shape ``[D, C]``."""
    if not class_names:
        raise ValueError("class_names must not be empty")
    if not templates:
        raise ValueError("templates must not be empty")

    device = torch.device(device)
    model.eval()
    prototypes = []
    for class_name in class_names:
        prompts = [_format_prompt(template, class_name) for template in templates]
        features = _encode_text(model, tokenizer, prompts, device, max_text_length)
        features = _require_feature_matrix(features, "Text encoder")
        features = F.normalize(features, dim=-1)
        prototypes.append(F.normalize(features.mean(dim=0), dim=0))
    return torch.stack(prototypes, dim=1)


@dataclass(frozen=True)
class CLIPZeroShotResult:
    """Zero-shot rankings; accuracy is a fraction in ``[0, 1]``."""

    rankings: torch.Tensor
    labels: torch.Tensor

    @property
    def predictions(self) -> torch.Tensor:
        return self.rankings[:, 0]

    def topk_accuracy(self, k: int = 1) -> float:
        if k <= 0 or k > self.rankings.shape[1]:
            raise ValueError(f"k must be in [1, {self.rankings.shape[1]}]")
        hits = (self.rankings[:, :k] == self.labels[:, None]).any(dim=1)
        return hits.float().mean().item()


class CLIPZeroShotEvaluator:
    """Run zero-shot classification using a supplied CLIP model and prompts."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        class_names: Sequence[str],
        templates: Sequence[PromptTemplate],
        *,
        device: Optional[Union[str, torch.device]] = None,
        max_text_length: Optional[int] = 77,
    ):
        self.model = model
        self.class_names = tuple(class_names)
        self.device = torch.device(device or self._model_device(model))
        self.prototypes = build_clip_text_prototypes(
            model,
            tokenizer,
            self.class_names,
            templates,
            self.device,
            max_text_length=max_text_length,
        )

    @staticmethod
    def _model_device(model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _split_batch(batch):
        if isinstance(batch, Mapping):
            label_key = "labels" if "labels" in batch else "label"
            if "pixel_values" not in batch or label_key not in batch:
                raise ValueError("Mapping batches must contain pixel_values and label/labels")
            return batch["pixel_values"], batch[label_key]
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("Batches must be (pixel_values, labels) pairs or matching mappings")

    @torch.inference_mode()
    def evaluate(
        self,
        dataset: Union[Dataset, DataLoader],
        *,
        topk: Sequence[int] = (1,),
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: Optional[bool] = None,
        collate_fn=None,
    ) -> CLIPZeroShotResult:
        """Predict on preprocessed data without loading models or checkpoints."""
        topk = tuple(int(k) for k in topk)
        if not topk or min(topk) <= 0 or max(topk) > len(self.class_names):
            raise ValueError("topk must be nonempty with values between 1 and the class count")
        if isinstance(dataset, DataLoader):
            loader = dataset
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(
                    self.device.type == "cuda" if pin_memory is None else pin_memory
                ),
                collate_fn=collate_fn,
            )

        self.model.eval()
        rankings, labels = [], []
        for batch in loader:
            pixel_values, targets = self._split_batch(batch)
            pixel_values = pixel_values.to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            features = _require_feature_matrix(
                _encode_image(self.model, pixel_values),
                "Image encoder",
            )
            features = F.normalize(features, dim=-1)
            ranks = (features @ self.prototypes).topk(max(topk), dim=1).indices
            rankings.append(ranks.cpu())
            labels.append(torch.as_tensor(targets).long().cpu().reshape(-1))

        if not rankings:
            raise ValueError("The evaluation dataset must not be empty")
        rankings = torch.cat(rankings)
        labels = torch.cat(labels)
        if len(rankings) != len(labels):
            raise ValueError("Image and label counts do not match")
        return CLIPZeroShotResult(rankings=rankings, labels=labels)
