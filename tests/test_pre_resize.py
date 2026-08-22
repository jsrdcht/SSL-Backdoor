from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import warnings

import pytest
import torch
import yaml
from PIL import Image
from torchvision import transforms

from ssl_backdoor.datasets.dataset import (
    CTRLTrainDataset,
    FileListDataset,
    OnlineUniversalPoisonedValDataset,
    SSLBackdoorTrainDataset,
)
from ssl_backdoor.datasets.pre_resize import pre_resize_image, resolve_pre_resize


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, (False, None)),
        ({"pre_resize": True, "pre_resize_size": 32}, (True, 32)),
        ({"pre_resize": False, "pre_resize_size": 32}, (False, 32)),
    ],
)
def test_resolve_pre_resize_accepts_canonical_config(config, expected):
    assert resolve_pre_resize(config) == expected


def test_resolve_pre_resize_accepts_legacy_config_with_warning():
    with pytest.warns(FutureWarning, match="deprecated"):
        assert resolve_pre_resize({"trigger": {"pre_resize": 224}}) == (True, 224)


@pytest.mark.parametrize("legacy_value", [False, None])
def test_resolve_pre_resize_accepts_disabled_legacy_config(legacy_value):
    with pytest.warns(FutureWarning, match="deprecated"):
        assert resolve_pre_resize({"trigger": {"pre_resize": legacy_value}}) == (
            False,
            None,
        )


def test_resolve_pre_resize_accepts_matching_new_and_legacy_config():
    config = {
        "pre_resize": True,
        "pre_resize_size": [7, 5],
        "trigger": {"pre_resize": (7, 5)},
    }
    with pytest.warns(FutureWarning, match="ignored"):
        assert resolve_pre_resize(config) == (True, [7, 5])


@pytest.mark.parametrize(
    ("config", "error"),
    [
        ({"pre_resize": True}, ValueError),
        ({"pre_resize": 224, "pre_resize_size": 224}, TypeError),
        ({"data": {"pre_resize": 224}}, ValueError),
        (
            {
                "pre_resize": True,
                "pre_resize_size": 224,
                "trigger": {"pre_resize": 32},
            },
            ValueError,
        ),
    ],
)
def test_resolve_pre_resize_rejects_invalid_config(config, error):
    with pytest.raises(error):
        resolve_pre_resize(config)


def test_pre_resize_image_supports_pil_and_tensor():
    image = Image.new("RGB", (12, 8))
    tensor = torch.zeros(3, 8, 12)

    assert pre_resize_image(image, True, [7, 5]).size == (7, 5)
    assert pre_resize_image(image, True, 6).size == (6, 6)
    assert pre_resize_image(tensor, True, [7, 5]).shape == (3, 5, 7)
    assert pre_resize_image(image, False, None) is image


def test_pre_resize_image_uses_bilinear_interpolation():
    image = Image.new("L", (3, 2))
    image.putdata([0, 32, 64, 128, 192, 255])
    image = image.convert("RGB")
    bilinear = getattr(Image, "Resampling", Image).BILINEAR

    actual = pre_resize_image(image, True, [5, 4])
    expected = image.resize((5, 4), bilinear)

    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "args",
    [
        {"pre_resize": True, "pre_resize_size": [7, 5]},
        Namespace(pre_resize=True, pre_resize_size=[7, 5]),
    ],
)
def test_file_list_dataset_pre_resizes_before_transform(tmp_path, args):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (12, 8)).save(image_path)
    filelist = tmp_path / "images.txt"
    filelist.write_text(f"{image_path} 3\n", encoding="utf-8")

    dataset = FileListDataset(args, filelist, transform=lambda image: image.size)

    assert dataset[0] == ((7, 5), 3)


def _online_args():
    return Namespace(
        dataset="cifar10",
        return_attack_target=False,
        attack_target=0,
        attack_algorithm="clean",
        trigger_path=None,
        trigger_size=None,
        trigger_insert="patch",
        position="random",
        alpha=1.0,
        pre_resize=True,
        pre_resize_size=[10, 6],
    )


def test_online_and_pre_injected_datasets_share_resize_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (12, 8)).save(image_path)
    filelist = tmp_path / "images.txt"
    filelist.write_text(f"{image_path} 3\n", encoding="utf-8")
    transform = transforms.Compose(
        [transforms.Resize((5, 7)), transforms.Lambda(lambda image: image.size)]
    )
    trigger_sizes = []

    def record_poison(_self, image):
        trigger_sizes.append(image.size)
        return image

    monkeypatch.setattr(
        OnlineUniversalPoisonedValDataset, "apply_poison", record_poison
    )

    online = OnlineUniversalPoisonedValDataset(
        _online_args(), filelist, transform, pre_inject_mode=False
    )
    pre_injected = OnlineUniversalPoisonedValDataset(
        _online_args(), filelist, transform, pre_inject_mode=True
    )

    assert online[0] == ((7, 5), 3)
    assert pre_injected[0] == ((7, 5), 3)
    assert trigger_sizes == [(10, 6), (10, 6)]


def test_pre_injected_dataset_runs_geometric_transform_per_access(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (12, 8)).save(image_path)
    filelist = tmp_path / "images.txt"
    filelist.write_text(f"{image_path} 3\n", encoding="utf-8")
    crop = transforms.RandomResizedCrop((5, 7))
    calls = []

    def record_crop(image):
        calls.append(image.size)
        return image

    monkeypatch.setattr(crop, "forward", record_crop)
    dataset = OnlineUniversalPoisonedValDataset(
        _online_args(), filelist, transforms.Compose([crop]), pre_inject_mode=True
    )

    dataset[0]
    dataset[0]

    assert calls == [(10, 6), (10, 6)]


def test_training_datasets_resize_before_trigger(tmp_path, monkeypatch):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (12, 8)).save(image_path)
    trigger_sizes = []

    def record_trigger(image, *_args, **_kwargs):
        trigger_sizes.append(image.size)
        return image

    monkeypatch.setattr(
        "ssl_backdoor.datasets.dataset.add_watermark", record_trigger
    )
    dataset = object.__new__(SSLBackdoorTrainDataset)
    dataset.pre_resize = True
    dataset.pre_resize_size = [7, 5]
    dataset.save_poisons = True
    dataset.trigger_size = 2
    dataset.position = "center"
    dataset.location_min = 0.0
    dataset.location_max = 1.0
    dataset.alpha = 1.0
    dataset.trigger_insert = "patch"

    dataset.apply_poison(str(image_path), "unused.png")

    ctrl_dataset = object.__new__(CTRLTrainDataset)
    ctrl_dataset.pre_resize = True
    ctrl_dataset.pre_resize_size = [7, 5]
    ctrl_dataset.save_poisons = True
    ctrl_dataset.agent = SimpleNamespace(apply_poison=record_trigger)
    ctrl_dataset.apply_poison(str(image_path), "unused.png")

    assert trigger_sizes == [(7, 5), (7, 5)]


def test_repository_configs_use_canonical_pre_resize_layout():
    config_root = Path(__file__).parents[1] / "configs"
    for path in [*config_root.rglob("*.yaml"), *config_root.rglob("*.yml")]:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert isinstance(config, dict), path
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_pre_resize(config)
        assert not caught, path
