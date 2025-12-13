"""攻击/投毒相关实现（从 datasets 层拆分出来）。

该子包用于容纳各类 poisoning agent、生成器网络以及 CorruptEncoder 辅助函数。
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
