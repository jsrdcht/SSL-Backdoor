from pathlib import Path

import pytest
import torch
from PIL import Image

from ssl_backdoor.defenses.image_trigger import apply_static_trigger
from ssl_backdoor.defenses.subspace_detection.detector import SubspaceDetector
from ssl_backdoor.defenses.subspace_detection.text_variants import TextVariantBank


def test_detector_fit_and_score_are_deterministic():
    generator = torch.Generator().manual_seed(7)
    text_features = torch.randn(10, 8, generator=generator)
    images = torch.randn(3, 8, generator=generator)

    first = SubspaceDetector(seed=11).fit_class(text_features)
    second = SubspaceDetector(seed=11).fit_class(text_features)

    torch.testing.assert_close(first, second)
    assert first.shape == (15, 8)
    assert SubspaceDetector.score(images, first).shape == (3,)


def test_detector_validates_sampling_configuration():
    with pytest.raises(ValueError, match="num_augmented"):
        SubspaceDetector(num_augmented=91).fit_class(torch.randn(10, 4))
    with pytest.raises(ValueError, match="image_features"):
        SubspaceDetector.score(torch.randn(4), torch.randn(3, 4))


def test_text_resources_are_aligned():
    root = Path(__file__).resolve().parents[1] / "assets/subspace_detection_resources"
    bank = TextVariantBank(
        root / "imagenet1k_descriptions.csv",
        root / "imagenet1k_arabic_classes.txt",
    )

    assert len(bank) == 1000
    assert bank.classes[954] == "banana"
    assert len(bank.texts(954)) == 10
    assert all(bank.texts(954))


def test_static_patch_trigger_and_alias(tmp_path):
    trigger_path = tmp_path / "trigger.png"
    Image.new("RGB", (2, 2), "white").save(trigger_path)
    image = Image.new("RGB", (8, 8), "black")
    config = {
        "attack_algorithm": "sslbkd",
        "trigger_path": str(trigger_path),
        "trigger_size": 2,
        "position": "badnet",
        "alpha": 1.0,
    }

    result = apply_static_trigger(image, config)

    assert result.size == image.size
    assert result.getpixel((5, 5)) == (255, 255, 255)
    assert result.getpixel((0, 0)) == (0, 0, 0)


def test_static_trigger_supports_wanet_through_compatibility_import():
    image = Image.new("RGB", (8, 8))
    result = apply_static_trigger(image, {"attack_algorithm": "wanet"})
    assert result.size == image.size
