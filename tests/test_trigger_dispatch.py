from argparse import Namespace

import pytest
from PIL import Image

from ssl_backdoor.attacks.clip_backdoor.utils import build_trigger_args
from ssl_backdoor.datasets.attacker.triggers import (
    _resolve_trigger_name,
    apply_static_trigger,
)
from ssl_backdoor.datasets.attacker.agent import CTRLPoisoningAgent
from ssl_backdoor.defenses.image_trigger import apply_static_trigger as defense_trigger


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"attack_algorithm": "badclip"}, "patch"),
        ({"attack_algorithm": "badnet"}, "patch"),
        ({"attack_algorithm": "sslbkd"}, "patch"),
        ({"trigger_insert": "badclip"}, "patch"),
        ({"trigger_insert": "badnet"}, "patch"),
        ({"trigger_insert": "sslbkd"}, "patch"),
        ({"attack_algorithm": "sslbkd", "trigger_insert": "blend"}, "blend"),
        ({"attack_algorithm": "ctrl", "trigger_insert": "patch"}, "ctrl"),
        ({"attack_algorithm": "refool_ghost", "trigger_insert": "patch"}, "refool_ghost"),
    ],
)
def test_trigger_selection_precedence(args, expected):
    assert _resolve_trigger_name(Namespace(**args)) == expected


def test_build_trigger_args_preserves_trigger_insert():
    assert build_trigger_args({"trigger_insert": "blend", "alpha": 0.3}) == {
        "trigger_insert": "blend",
        "alpha": 0.3,
    }


def test_online_alias_applies_patch(tmp_path):
    trigger_path = tmp_path / "trigger.png"
    Image.new("RGB", (2, 2), "white").save(trigger_path)
    image = Image.new("RGB", (8, 8), "black")

    result = apply_static_trigger(
        image,
        {
            "attack_algorithm": "badclip",
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
            "attack_algorithm": "sslbkd",
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
            "lindct": True,
        }
    )

    assert agent.channel_list == [0]
    assert agent.window_size == 16
    assert agent.pos_list == [(1, 2)]
    assert agent.magnitude == 7
    assert agent.lindct is True


def test_defense_import_is_compatibility_alias():
    assert defense_trigger is apply_static_trigger
