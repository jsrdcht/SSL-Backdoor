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

__all__ = [
    "CTRLPoisoningAgent",
    "AdaptivePoisoningAgent",
    "BadEncoderPoisoningAgent",
    "BadCLIPPoisoningAgent",
    "ExternalServicePoisoningAgent",
]
