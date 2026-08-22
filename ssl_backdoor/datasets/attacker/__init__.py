"""Poisoning-related implementations moved out from the datasets layer.

This subpackage contains poisoning agents, generator networks, and CorruptEncoder helper utilities.
"""

from .agent import (
    CTRLPoisoningAgent,
    AdaptivePoisoningAgent,
    BadEncoderPoisoningAgent,
    BadCLIPPoisoningAgent,
    ExternalServicePoisoningAgent,
)
from .triggers import (
    apply_refool_trigger,
    apply_sig_trigger,
    apply_static_trigger,
    apply_wanet_trigger,
    blend_refool_images,
)
from .trigger_templates import TRIGGER_PARAM_TEMPLATES

__all__ = [
    "CTRLPoisoningAgent",
    "AdaptivePoisoningAgent",
    "BadEncoderPoisoningAgent",
    "BadCLIPPoisoningAgent",
    "ExternalServicePoisoningAgent",
    "apply_static_trigger",
    "apply_refool_trigger",
    "apply_sig_trigger",
    "apply_wanet_trigger",
    "blend_refool_images",
    "TRIGGER_PARAM_TEMPLATES",
]
