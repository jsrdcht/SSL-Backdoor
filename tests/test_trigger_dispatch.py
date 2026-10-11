from argparse import Namespace

import pytest
from PIL import Image

from ssl_backdoor.attacks.clip_backdoor.utils import build_trigger_args
from ssl_backdoor.datasets.attacker.triggers import (
    _resolve_trigger_name,
    apply_static_trigger,
)
from ssl_backdoor.datasets.attacker.agent import CTRLPoisoningAgent
from ssl_backdoor.datasets.dataset import OnlineUniversalPoisonedValDataset
from ssl_backdoor.defenses.image_trigger import apply_static_trigger as defense_trigger


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"attack_algorithm": "na", "trigger_insert": "blend"}, "blend"),
        ({"attack_algorithm": "sslbkd", "trigger_insert": "blend"}, "blend"),
        ({"attack_algorithm": "patch", "trigger_insert": "patch"}, "patch"),
        ({"trigger_insert": "ctrl"}, "ctrl"),
    ],
)
def test_trigger_selection_uses_canonical_fields(args, expected):
    assert _resolve_trigger_name(Namespace(**args)) == expected


def test_trigger_selection_rejects_conflicting_canonical_fields():
    with pytest.raises(ValueError, match="Conflicting trigger selectors"):
        _resolve_trigger_name(
            Namespace(attack_algorithm="ctrl", trigger_insert="patch")
        )


@pytest.mark.parametrize(
    "args",
    [
        {"attack_algorithm": "badclip"},
        {"trigger_insert": "badclip"},
        {"attack_algorithm": "refool_ghost", "trigger_insert": "patch"},
        {"trigger_type": "blend"},
    ],
)
def test_trigger_selection_rejects_noncanonical_names(args):
    with pytest.raises(ValueError):
        _resolve_trigger_name(Namespace(**args))


def test_build_trigger_args_preserves_canonical_config():
    config = {
        "trigger_insert": "blend",
        "trigger_path": "trigger.png",
        "alpha": 0.3,
    }
    assert build_trigger_args(config) == config


@pytest.mark.parametrize(
    "config",
    [
        {"attack_algorithm": "sslbkd", "trigger_path": "trigger.png"},
        {
            "trigger_insert": "sig",
            "sig_amplitude": 1.0,
        },
    ],
)
def test_build_trigger_args_rejects_legacy_fields(config):
    with pytest.raises(ValueError):
        build_trigger_args(config)


def test_static_patch_requires_trigger_path():
    with pytest.raises(ValueError, match="trigger_path is required"):
        apply_static_trigger(
            Image.new("RGB", (8, 8)),
            {"trigger_insert": "patch"},
        )


@pytest.mark.parametrize(
    "args,field",
    [
        (
            Namespace(trigger_insert="sig", trigger_mode="sig"),
            "trigger_mode",
        ),
        (
            Namespace(trigger_insert="sig", sig_freq=2.0),
            "sig_freq",
        ),
    ],
)
def test_namespace_trigger_args_reject_removed_aliases(args, field):
    with pytest.raises(ValueError, match=field):
        apply_static_trigger(Image.new("RGB", (8, 8)), args)


def test_canonical_patch_trigger(tmp_path):
    trigger_path = tmp_path / "trigger.png"
    Image.new("RGB", (2, 2), "white").save(trigger_path)
    image = Image.new("RGB", (8, 8), "black")

    result = apply_static_trigger(
        image,
        {
            "trigger_insert": "patch",
            "trigger_path": str(trigger_path),
            "trigger_size": 2,
            "position": "badnet",
        },
    )

    assert result.size == image.size
    assert result.getpixel((5, 5)) == (255, 255, 255)
    assert result.getpixel((0, 0)) == (0, 0, 0)


def test_center_patch_position(tmp_path):
    trigger_path = tmp_path / "trigger.png"
    Image.new("RGB", (2, 2), "white").save(trigger_path)

    result = apply_static_trigger(
        Image.new("RGB", (8, 8), "black"),
        {
            "trigger_insert": "patch",
            "trigger_path": str(trigger_path),
            "trigger_size": 2,
            "position": "center",
        },
    )

    assert result.getpixel((3, 3)) == (255, 255, 255)
    assert result.getpixel((5, 5)) == (0, 0, 0)


def test_blend_trigger_uses_configured_alpha(tmp_path):
    trigger_path = tmp_path / "trigger.png"
    Image.new("RGB", (2, 2), "white").save(trigger_path)

    result = apply_static_trigger(
        Image.new("RGB", (4, 4), "black"),
        {
            "trigger_insert": "blend",
            "trigger_path": str(trigger_path),
            "alpha": 0.25,
        },
    )

    assert result.getpixel((0, 0)) == (63, 63, 63)


def test_ctrl_agent_reads_mapping_parameters():
    agent = CTRLPoisoningAgent(
        {
            "channel_list": [0],
            "window_size": 16,
            "pos_list": [[1, 2]],
            "attack_magnitude": 7,
        }
    )

    assert agent.channel_list == [0]
    assert agent.window_size == 16
    assert agent.pos_list == [(1, 2)]
    assert agent.magnitude == 7


def test_ctrl_rejects_removed_lindct_option():
    with pytest.raises(ValueError, match="lindct"):
        apply_static_trigger(
            Image.new("RGB", (32, 32)),
            {"trigger_insert": "ctrl", "lindct": True},
        )
    with pytest.raises(ValueError, match="lindct"):
        CTRLPoisoningAgent({"lindct": False})


def test_algorithm_mode_is_not_treated_as_trigger_selector():
    result = apply_static_trigger(
        Image.new("RGB", (8, 8)),
        Namespace(
            attack_algorithm="drupe",
            trigger_insert="sig",
            mode="drupe",
        ),
    )
    assert result.size == (8, 8)


@pytest.mark.parametrize("trigger_insert", ["ctrl", "wanet"])
def test_mutable_trigger_config_allows_runtime_cache(trigger_insert):
    config = {"trigger_insert": trigger_insert}
    image = Image.new("RGB", (32, 32))

    assert apply_static_trigger(image, config).size == image.size
    assert apply_static_trigger(image, config).size == image.size


def test_universal_dataset_rejects_agent_trigger_conflict(tmp_path):
    file_list = tmp_path / "test.txt"
    file_list.write_text("image.png 0\n", encoding="utf-8")
    args = Namespace(
        attack_algorithm="ctrl",
        trigger_insert="patch",
        attack_target=0,
        dataset="cifar10",
    )

    with pytest.raises(ValueError, match="requires trigger_insert='ctrl'"):
        OnlineUniversalPoisonedValDataset(args, file_list, transform=None)


def test_universal_dataset_uses_shared_static_dispatcher(tmp_path, monkeypatch):
    file_list = tmp_path / "test.txt"
    file_list.write_text("image.png 0\n", encoding="utf-8")
    args = Namespace(
        attack_algorithm="sslbkd",
        trigger_insert="patch",
        trigger_path="trigger.png",
        trigger_size=2,
        attack_target=0,
        dataset="cifar10",
    )
    calls = []

    def record_trigger(image, trigger_args):
        calls.append(trigger_args)
        return image

    monkeypatch.setattr(
        "ssl_backdoor.datasets.dataset.apply_static_trigger", record_trigger
    )
    dataset = OnlineUniversalPoisonedValDataset(args, file_list, transform=None)
    image = Image.new("RGB", (8, 8))

    assert dataset.apply_poison(image) is image
    assert calls == [args]


def test_defense_import_is_compatibility_alias():
    assert defense_trigger is apply_static_trigger
