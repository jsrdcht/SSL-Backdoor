"""Core Subspace Detection algorithm."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SubspaceDetector:
    """Build class-local text subspaces and score samples as in Algorithm 1."""

    def __init__(
        self,
        variance_ratio: float = 0.95,
        num_components: int = 3,
        num_augmented: int = 90,
        num_detection_samples: int = 15,
        eta: float = 1e-3,
        cosine_ratio: tuple[float, float] = (0.85, 1.0),
        max_positive_trials: int = 128,
        seed: int = 42,
    ):
        if not 0 < variance_ratio <= 1:
            raise ValueError("variance_ratio must be in (0, 1]")
        if min(num_components, num_augmented, num_detection_samples, max_positive_trials) <= 0:
            raise ValueError("sampling counts must be positive integers")
        if not 0 <= cosine_ratio[0] < cosine_ratio[1] <= 1:
            raise ValueError("cosine_ratio must be an increasing interval within [0, 1]")
        self.variance_ratio = variance_ratio
        self.num_components = num_components
        self.num_augmented = num_augmented
        self.num_detection_samples = num_detection_samples
        self.eta = eta
        self.cosine_ratio = cosine_ratio
        self.max_positive_trials = max_positive_trials
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def _uniform(self) -> float:
        return torch.rand((), generator=self.generator).item()

    def _normal(self, size: int, like: torch.Tensor) -> torch.Tensor:
        return torch.randn(size, generator=self.generator, dtype=like.dtype).to(like.device)

    def _positive_sample(
        self,
        variant: torch.Tensor,
        original: torch.Tensor,
        mean: torch.Tensor,
        basis: torch.Tensor,
        original_feature: torch.Tensor,
        variant_feature: torch.Tensor,
    ) -> torch.Tensor:
        denominator = F.cosine_similarity(
            variant_feature[None], original_feature[None]
        ).clamp_min(1e-8)
        best = variant
        for _ in range(self.max_positive_trials):
            candidate = variant + self._uniform() * (variant - original)
            restored = mean + candidate @ basis
            ratio = F.cosine_similarity(restored[None], original_feature[None]) / denominator
            best = candidate
            if self.cosine_ratio[0] < ratio.item() < self.cosine_ratio[1]:
                break
        return best

    @torch.no_grad()
    def fit_class(self, text_features: torch.Tensor) -> torch.Tensor:
        """Map ``[original text, 9 variants]`` to sampled features ``[ns, dim]``."""
        if text_features.ndim != 2 or len(text_features) < 2:
            raise ValueError("text_features must be a 2D tensor with at least two rows")
        num_variants = len(text_features) - 1
        if self.num_augmented % num_variants:
            raise ValueError("num_augmented must be divisible by the number of text variants")

        features = text_features.float()
        mean = features.mean(0)
        centered = features - mean
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        explained = singular_values.square()
        total_variance = explained.sum()
        if total_variance <= torch.finfo(features.dtype).eps:
            return mean.expand(self.num_detection_samples, -1).to(text_features.dtype)

        cumulative = explained.cumsum(0) / total_variance
        k = min(int(torch.searchsorted(cumulative, self.variance_ratio).item() + 1), len(vh))
        basis = vh[:k]
        projected = centered @ basis.T
        original, variants = projected[0], projected[1:]
        per_variant = self.num_augmented // num_variants

        components = []
        for _ in range(self.num_components):
            augmented = [*variants]
            for variant, variant_feature in zip(variants, features[1:]):
                augmented.extend(
                    self._positive_sample(
                        variant, original, mean, basis, features[0], variant_feature
                    )
                    for _ in range(per_variant)
                )
            points = torch.stack(augmented)
            component_mean = points.mean(0)
            components.append((component_mean, points - component_mean))

        samples = []
        for _ in range(self.num_detection_samples):
            component = min(int(self._uniform() * len(components)), len(components) - 1)
            component_mean, deviations = components[component]
            projected_sample = component_mean + self.eta * deviations.T @ self._normal(
                len(deviations), deviations
            )
            samples.append(mean + projected_sample @ basis)
        return torch.stack(samples).to(text_features.dtype)

    @staticmethod
    def score(image_features: torch.Tensor, sampled_text_features: torch.Tensor) -> torch.Tensor:
        """Compute mean squared distance to the class-specific region of interest."""
        if image_features.ndim != 2:
            raise ValueError("image_features must have shape [batch, dim]")
        if sampled_text_features.ndim == 2:
            sampled_text_features = sampled_text_features.unsqueeze(0)
        if sampled_text_features.ndim != 3:
            raise ValueError(
                "sampled_text_features must have shape [samples, dim] or [batch, samples, dim]"
            )
        if sampled_text_features.shape[0] not in (1, len(image_features)):
            raise ValueError("text distribution batch size must be 1 or match the image batch")
        return (image_features[:, None] - sampled_text_features).square().sum(-1).mean(-1)
