"""Deterministic ImageNet reference and mixed evaluation sets for BDetCLIP."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ssl_backdoor.datasets.attacker.triggers import apply_static_trigger
from ssl_backdoor.datasets.attacker.trigger_templates import validate_trigger_config
from ssl_backdoor.datasets.pre_resize import pre_resize_image


def split_samples(samples, *, reference_samples, evaluation_samples, poison_ratio, target, seed):
    required = reference_samples + evaluation_samples
    if required > len(samples):
        raise ValueError(f"not enough samples: required {required}, found {len(samples)}")
    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(samples))[:required]
    reference = [samples[index] for index in selected[:reference_samples]]
    evaluation = [samples[index] for index in selected[reference_samples:]]
    candidates = [index for index, (_, label) in enumerate(evaluation) if label != target]
    num_poison = round(evaluation_samples * poison_ratio)
    if num_poison <= 0 or num_poison >= evaluation_samples:
        raise ValueError("poison_ratio must produce both clean and poisoned samples")
    if num_poison > len(candidates):
        raise ValueError("not enough non-target samples for the requested poison ratio")
    poisoned = set(rng.choice(candidates, size=num_poison, replace=False).tolist())
    return reference, evaluation, poisoned


class MixedTriggeredDataset(Dataset):
    def __init__(
        self,
        samples,
        poisoned_indices,
        process_image,
        trigger,
        seed,
        pre_resize=False,
        pre_resize_size=None,
    ):
        self.samples = list(samples)
        self.poisoned_indices = set(poisoned_indices)
        self.process_image = process_image
        self.trigger = dict(trigger)
        self.trigger.pop("trigger_interpolation", None)
        validate_trigger_config(self.trigger, require_path=True)
        self.seed = seed
        self.pre_resize = pre_resize
        self.pre_resize_size = pre_resize_size
        self.trigger_image = None
        interpolation = trigger.get("trigger_interpolation")
        if interpolation:
            modes = {
                "nearest": Image.Resampling.NEAREST,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
            }
            if interpolation not in modes:
                raise ValueError(f"unsupported trigger_interpolation: {interpolation}")
            size = int(trigger["trigger_size"])
            self.trigger_image = Image.open(trigger["trigger_path"]).convert("RGB").resize(
                (size, size), modes[interpolation]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = pre_resize_image(image, self.pre_resize, self.pre_resize_size)
        is_poisoned = index in self.poisoned_indices
        if is_poisoned:
            item_seed = self.seed + index
            random.seed(item_seed)
            np.random.seed(item_seed % (2**32))
            torch.manual_seed(item_seed)
            trigger = self.trigger_image or self.trigger.get("trigger_path")
            image = apply_static_trigger(image, self.trigger, trigger)
        return self.process_image(image), label, str(path), is_poisoned
