"""Song et al.'s test-time Subspace Detection defense for CLIP."""

from .detector import SubspaceDetector
from .evaluation import run_subspace_detection
from .text_variants import TextVariantBank

__all__ = ["SubspaceDetector", "TextVariantBank", "run_subspace_detection"]
