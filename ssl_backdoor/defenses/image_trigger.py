"""Minimal static image-trigger support shared by test-time defenses."""

from __future__ import annotations

from ssl_backdoor.datasets.utils import add_watermark


_PATCH_ALIASES = {"patch", "badclip", "badnet", "sslbkd"}


def apply_static_trigger(image, config: dict, trigger=None):
    """Apply a patch or blend trigger described by a defense config."""
    algorithm = str(config.get("attack_algorithm", "patch")).lower()
    if algorithm in _PATCH_ALIASES:
        mode = "patch"
    elif algorithm == "blend":
        mode = "blend"
    else:
        raise ValueError(
            f"Unsupported online trigger {algorithm!r}; provide data.poisoned_csv "
            "for attacks other than patch/blend"
        )

    return add_watermark(
        image,
        trigger or config.get("trigger_path"),
        watermark_width=int(config.get("trigger_size", 50)),
        position=config.get("position", "random"),
        location_min=float(config.get("location_min", 0.25)),
        location_max=float(config.get("location_max", 0.75)),
        alpha=float(config.get("alpha", 1.0)),
        mode=mode,
    )
