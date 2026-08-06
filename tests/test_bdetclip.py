from pathlib import Path

import numpy as np
import pytest
import torch

from ssl_backdoor.defenses.bdetclip.data import split_samples
from ssl_backdoor.defenses.bdetclip.evaluation import (
    aggregate,
    build_direction,
    detection_metrics,
    score_features,
)
from ssl_backdoor.defenses.bdetclip.prompts import PromptBank


def test_optimized_score_matches_classwise_formula():
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(5, 4, generator=generator)
    benign = torch.nn.functional.normalize(torch.randn(3, 4, generator=generator), dim=-1)
    malignant = torch.nn.functional.normalize(torch.randn(3, 4, generator=generator), dim=-1)
    images = torch.nn.functional.normalize(images, dim=-1)
    expected = ((images @ benign.T) - (images @ malignant.T)).sum(1)
    actual = score_features(images, build_direction(benign, malignant))
    torch.testing.assert_close(actual, expected)


def test_aggregate_normalizes_group_means():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [2.0, 0.0]])
    result = aggregate(features, 2, 2)
    torch.testing.assert_close(result.norm(dim=1), torch.ones(2))
    with pytest.raises(ValueError):
        aggregate(features, 3, 2)


def test_reference_threshold_and_score_direction():
    metrics = detection_metrics(
        omega=np.array([1.2, 1.1, 0.2, 0.1]),
        poisoned=np.array([0, 0, 1, 1]),
        reference_omega=np.array([1.0, 0.9]),
    )
    assert metrics["threshold"] == pytest.approx(0.9)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["threshold_f1"] == pytest.approx(1.0)


def test_split_is_deterministic_disjoint_and_excludes_target():
    samples = [(f"image-{i}", i % 10) for i in range(100)]
    first = split_samples(
        samples,
        reference_samples=20,
        evaluation_samples=50,
        poison_ratio=0.3,
        target=4,
        seed=42,
    )
    second = split_samples(
        samples,
        reference_samples=20,
        evaluation_samples=50,
        poison_ratio=0.3,
        target=4,
        seed=42,
    )
    assert first == second
    reference, evaluation, poisoned = first
    assert set(reference).isdisjoint(evaluation)
    assert len(poisoned) == 15
    assert all(evaluation[index][1] != 4 for index in poisoned)


def test_official_prompt_resources_are_aligned():
    root = Path(__file__).resolve().parents[1]
    bank = PromptBank.load(
        root / "assets/imagenet/classes.py",
        root / "assets/bdetclip/Benign_Imagenet1k.json",
        root / "assets/bdetclip/Malignant_Imagenet1k.txt",
    )
    assert len(bank.classes) == 1000
    assert len(bank.templates) == 80
    assert all(len(items) == 7 for items in bank.benign)
    assert bank.classes[954] == "banana"
