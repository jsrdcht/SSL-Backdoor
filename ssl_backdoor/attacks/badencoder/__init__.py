"""
BadEncoder攻击模块

BadEncoder: 一种针对自监督学习编码器的后门攻击实现
"""

from .badencoder import run_badencoder
from .datasets import get_poisoning_dataset, get_dataset_evaluation

__all__ = [
    'run_badencoder',
    'get_poisoning_dataset',
    'get_dataset_evaluation'
] 