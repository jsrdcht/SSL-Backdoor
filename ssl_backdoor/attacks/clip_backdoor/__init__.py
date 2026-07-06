"""CLIP backdoor: offline poison data generation and backdoor evaluation.

Poisoning happens in data preparation stage (generate poisoned CSV + poisoned images), training stage
reuses clean ``tools/train_clip.py`` / ``CLIPTrainer``, trainer is unaware of poisoning.
"""

from .poison_generator import generate_poison
from .caption_targets import build_target_caption, load_templates, resolve_target_index

__all__ = [
    "generate_poison",
    "build_target_caption",
    "load_templates",
    "resolve_target_index",
]
