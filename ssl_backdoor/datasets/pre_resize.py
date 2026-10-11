"""Shared optional image resizing before trigger injection and model preprocessing."""

import warnings
from collections.abc import Mapping

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


def _nested_pre_resize_keys(value, path=()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            current = (*path, str(key))
            if path and key in {"pre_resize", "pre_resize_size"}:
                yield ".".join(current)
            yield from _nested_pre_resize_keys(child, current)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _nested_pre_resize_keys(child, (*path, str(index)))


def _target_size(size):
    if isinstance(size, bool):
        raise TypeError("pre_resize_size must be an integer or a two-item sequence")
    if isinstance(size, int):
        dimensions = (size, size)
    elif isinstance(size, (list, tuple)) and len(size) == 2:
        dimensions = tuple(size)
    else:
        raise TypeError("pre_resize_size must be an integer or a two-item sequence")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
        raise TypeError("pre_resize_size dimensions must be integers")
    if any(value <= 0 for value in dimensions):
        raise ValueError("pre_resize_size dimensions must be positive")
    return dimensions


def resolve_pre_resize(config: Mapping) -> tuple[bool, object | None]:
    """Resolve the canonical top-level setting, accepting one legacy layout."""
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")

    nested_keys = set(_nested_pre_resize_keys(config))
    unsupported = nested_keys - {"trigger.pre_resize"}
    if unsupported:
        raise ValueError(
            "pre_resize and pre_resize_size must be top-level fields; "
            f"invalid nested fields: {sorted(unsupported)}"
        )

    trigger = config.get("trigger")
    has_legacy = isinstance(trigger, Mapping) and "pre_resize" in trigger
    legacy_size = trigger.get("pre_resize") if has_legacy else None
    legacy_enabled = legacy_size is not None and legacy_size is not False
    if legacy_enabled:
        _target_size(legacy_size)

    has_enabled = "pre_resize" in config
    has_size = "pre_resize_size" in config
    if has_enabled != has_size:
        raise ValueError("pre_resize and pre_resize_size must be configured together")

    if not has_enabled:
        if not has_legacy:
            return False, None
        warnings.warn(
            "trigger.pre_resize is deprecated; use top-level pre_resize and "
            "pre_resize_size instead",
            FutureWarning,
            stacklevel=2,
        )
        return (True, legacy_size) if legacy_enabled else (False, None)

    enabled = config["pre_resize"]
    size = config["pre_resize_size"]
    if not isinstance(enabled, bool):
        raise TypeError("pre_resize must be a boolean")
    if enabled and size is None:
        raise ValueError("pre_resize_size is required when pre_resize is enabled")
    if size is not None:
        _target_size(size)

    if has_legacy:
        if enabled != legacy_enabled or (
            enabled and _target_size(size) != _target_size(legacy_size)
        ):
            raise ValueError("top-level and legacy pre_resize settings conflict")
        warnings.warn(
            "trigger.pre_resize is deprecated and ignored in favor of the top-level setting",
            FutureWarning,
            stacklevel=2,
        )
    return enabled, size


def pre_resize_image(image, enabled=False, size=None):
    """Resize a PIL image or CHW tensor with bilinear interpolation."""
    if not isinstance(enabled, bool):
        raise TypeError("pre_resize must be a boolean")
    if not enabled:
        return image
    if size is None:
        raise ValueError("pre_resize_size is required when pre_resize is enabled")

    width, height = _target_size(size)
    if torch.is_tensor(image):
        return resize(
            image,
            [height, width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image or tensor")
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize((width, height), bilinear)
