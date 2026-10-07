from pathlib import Path

import pytest
import torch
from PIL import Image

from ssl_backdoor.defenses.image_trigger import apply_static_trigger
from ssl_backdoor.defenses.subspace_detection.data import (
    ImageSampleDataset,
    PairedTriggeredDataset,
)
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


def test_static_patch_trigger(tmp_path):
    trigger_path = tmp_path / "trigger.png"
    Image.new("RGB", (2, 2), "white").save(trigger_path)
    image = Image.new("RGB", (8, 8), "black")
    config = {
        "trigger_insert": "patch",
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
    result = apply_static_trigger(image, {"trigger_insert": "wanet"})
    assert result.size == image.size


def test_subspace_datasets_share_pre_resize(tmp_path, monkeypatch):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (12, 8)).save(image_path)
    samples = [(image_path, 2)]
    trigger_sizes = []

    def record_trigger(image, *_args, **_kwargs):
        trigger_sizes.append(image.size)
        return image

    monkeypatch.setattr(
        "ssl_backdoor.defenses.subspace_detection.data.apply_static_trigger",
        record_trigger,
    )
    resize_args = {"pre_resize": True, "pre_resize_size": [7, 5]}
    reference = ImageSampleDataset(samples, lambda image: image.size, **resize_args)
    paired = PairedTriggeredDataset(
        samples,
        lambda image: image.size,
        {"trigger_insert": "patch", "trigger_path": "unused.png"},
        seed=3,
        **resize_args,
    )

    assert reference[0][0] == (7, 5)
    assert paired[0][:3] == ((7, 5), (7, 5), 2)
    assert trigger_sizes == [(7, 5)]
