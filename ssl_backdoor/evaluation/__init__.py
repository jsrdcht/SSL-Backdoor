"""Shared evaluation components."""

from .clip_zero_shot import (
    CLIPZeroShotEvaluator,
    CLIPZeroShotResult,
    build_clip_text_prototypes,
)

__all__ = [
    "CLIPZeroShotEvaluator",
    "CLIPZeroShotResult",
    "build_clip_text_prototypes",
]
