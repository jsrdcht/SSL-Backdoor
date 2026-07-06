"""BadCLIP: Dual-Embedding Guided Backdoor Attack for Multimodal Contrastive Learning (CVPR 2024).

Key Features:
- Trigger Optimization: Optimizes trigger patterns via gradient descent to align with target semantics in embedding space
- Dual Constraints: Image-text alignment loss + triplet loss (trigger-target feature alignment)
- Defense Resistance: Triggers align with target visual features, making them hard to remove via fine-tuning
"""

from .trigger_optimizer import BadCLIPTriggerOptimizer

__all__ = ['BadCLIPTriggerOptimizer']
