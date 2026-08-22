import random
import copy
from types import SimpleNamespace

import cv2
import numpy as np
import scipy.stats as st
from PIL import Image

from ..utils import add_watermark, load_image
from .agent import AdaptivePoisoningAgent, CTRLPoisoningAgent
from .trigger_templates import PARAM_ALIASES, TRIGGER_ALIASES, TRIGGER_PARAM_TEMPLATES, trigger_defaults


def _get_arg(args, name, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _set_arg(args, name, value):
    if isinstance(args, dict):
        args[name] = value
    else:
        setattr(args, name, value)


def _has_value(value):
    if value is None:
        return False
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return len(value) > 0
    return True


def _build_trigger_params(args, trigger_name):
    params = SimpleNamespace(**copy.deepcopy(trigger_defaults(trigger_name)))
    for name in vars(params):
        for key in (name,) + PARAM_ALIASES.get(name, ()):
            value = _get_arg(args, key, None)
            if _has_value(value):
                setattr(params, name, value)
                break
    return params


def _resolve_trigger_name(args):
    attack_algorithm = str(_get_arg(args, "attack_algorithm", "")).lower()
    if attack_algorithm in TRIGGER_PARAM_TEMPLATES and attack_algorithm not in {
        "patch",
        "blend",
    }:
        return attack_algorithm

    if attack_algorithm.startswith("refool_") and attack_algorithm in TRIGGER_ALIASES:
        return attack_algorithm

    for key in ("trigger_insert", "trigger_type", "trigger_mode", "static_trigger", "mode"):
        value = _get_arg(args, key, None)
        if isinstance(value, bool):
            continue
        if value:
            trigger_name = str(value).lower()
            if trigger_name.startswith("refool_") and trigger_name in TRIGGER_ALIASES:
                return trigger_name
            return TRIGGER_ALIASES.get(trigger_name, trigger_name)

    if attack_algorithm in TRIGGER_ALIASES:
        return TRIGGER_ALIASES[attack_algorithm]
    if attack_algorithm in {"patch", "blend"}:
        return attack_algorithm

    return "patch"


def _resolve_trigger_path(args, trigger=None):
    if trigger is not None:
        return trigger
    for key in ("trigger_path", "reflection_path", "refool_reflection_path"):
        value = _get_arg(args, key, None)
        if value:
            return value
    raise ValueError("trigger_path is required for this trigger")


def _cached_agent(args, cache_name, agent_cls):
    agent = _get_arg(args, cache_name, None)
    if agent is None:
        agent = agent_cls(args)
        _set_arg(args, cache_name, agent)
    return agent


def apply_static_trigger(img, args, trigger=None):
    """Dispatch a PIL image to a static trigger implementation."""
    trigger_name = _resolve_trigger_name(args)
    params = _build_trigger_params(args, trigger_name)

    if trigger_name in {"patch", "blend"}:
        return apply_patch_or_blend_trigger(img, args, trigger, trigger_name, params)
    if trigger_name.startswith("refool"):
        return apply_refool_trigger(img, args, trigger, trigger_name, params)
    if trigger_name == "ctrl":
        return _cached_agent(args, "_ctrl_poisoning_agent", CTRLPoisoningAgent).apply_poison(_as_pil(img))
    if trigger_name == "sig":
        return apply_sig_trigger(img, args)
    if trigger_name == "wanet":
        return apply_wanet_trigger(img, args)
    if trigger_name == "blto":
        return _cached_agent(args, "_blto_poisoning_agent", AdaptivePoisoningAgent).apply_poison(img)

    raise ValueError(f"Unsupported trigger type: {trigger_name}")


def apply_patch_or_blend_trigger(img, args, trigger=None, mode=None, params=None):
    mode = mode or _resolve_trigger_name(args)
    if params is None:
        params = _build_trigger_params(args, mode)
    return add_watermark(
        img,
        _resolve_trigger_path(args, trigger),
        watermark_width=getattr(params, "trigger_size", None),
        position=getattr(params, "position", None),
        location_min=getattr(params, "location_min", None),
        location_max=getattr(params, "location_max", None),
        alpha=params.alpha,
        return_location=False,
        mode=mode,
    )


def _as_pil(img):
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    raise ValueError("img must be a PIL image, path, or numpy array")


def _to_rgb_array(img):
    return np.asarray(_as_pil(img).convert("RGB"), dtype=np.uint8)


def _alpha_t_from_params(params):
    alpha_t = params.alpha_t
    if alpha_t is not None:
        return float(alpha_t)

    alpha = params.alpha
    if alpha is None:
        return -1.0
    return 1.0 - float(alpha)


def apply_refool_trigger(img, args, trigger=None, mode=None, params=None):
    if params is None:
        params = _build_trigger_params(args, mode or _resolve_trigger_name(args))
    reflection = _resolve_trigger_path(args, trigger)
    img_pil = _as_pil(img).convert("RGB")
    refl_pil = load_image(reflection, mode="RGB")

    mode = (mode or _resolve_trigger_name(args)).lower()
    if "ghost" in mode:
        ghost_rate = 1.0
    elif "smooth" in mode or "blur" in mode:
        ghost_rate = 0.0
    else:
        ghost_rate = float(params.ghost_rate)

    max_image_size = params.refool_max_image_size
    if max_image_size is None:
        max_image_size = max(img_pil.size)

    offset = params.offset
    if offset is None:
        offset = (0, 0)

    blended, _, _ = blend_refool_images(
        _to_rgb_array(img_pil),
        _to_rgb_array(refl_pil),
        max_image_size=int(max_image_size),
        ghost_rate=ghost_rate,
        alpha_t=_alpha_t_from_params(params),
        offset=tuple(offset),
        sigma=float(params.sigma),
        ghost_alpha=float(params.ghost_alpha),
    )

    out = Image.fromarray(blended, mode="RGB")
    if bool(params.refool_preserve_size) and out.size != img_pil.size:
        out = out.resize(img_pil.size, Image.BILINEAR)
    return out


def blend_refool_images(img_t, img_r, max_image_size=560, ghost_rate=0.49, alpha_t=-1.0, offset=(0, 0), sigma=-1.0, ghost_alpha=-1.0):
    """Refool blend_images ported from DreamtaleCore/Refool scripts/insert_reflection.py."""
    t = np.float32(img_t) / 255.0
    r = np.float32(img_r) / 255.0
    h, w, _ = t.shape
    scale_ratio = float(max(h, w)) / float(max_image_size)
    w, h = (max_image_size, int(round(h / scale_ratio))) if w > h else (int(round(w / scale_ratio)), max_image_size)
    t = cv2.resize(t, (w, h), interpolation=cv2.INTER_CUBIC)
    r = cv2.resize(r, (w, h), interpolation=cv2.INTER_CUBIC)

    if alpha_t < 0:
        alpha_t = 1.0 - random.uniform(0.05, 0.45)

    if random.random() < ghost_rate:
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        if offset[0] == 0 and offset[1] == 0:
            offset = (random.randint(3, 8), random.randint(3, 8))

        r_1 = np.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)), "constant", constant_values=0)
        r_2 = np.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)), "constant", constant_values=(0, 0))
        if ghost_alpha < 0:
            ghost_alpha = abs(round(random.random()) - random.uniform(0.15, 0.5))

        ghost_r = r_1 * ghost_alpha + r_2 * (1.0 - ghost_alpha)
        ghost_r = cv2.resize(
            ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :],
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
        reflection_mask = ghost_r * (1.0 - alpha_t)
        blended = reflection_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1.0 / 2.2)
        ghost_r = np.clip(np.power(reflection_mask, 1.0 / 2.2), 0, 1)
        blended = np.clip(np.power(blended, 1.0 / 2.2), 0, 1)

        reflection_layer = np.uint8(ghost_r * 255)
        blended = np.uint8(blended * 255)
        transmission_layer = np.uint8(transmission_layer * 255)
    else:
        if sigma < 0:
            sigma = random.uniform(1, 5)

        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t

        att = 1.08 + np.random.random() / 10.0
        for i in range(3):
            mask_i = blend[:, :, i] > 1
            mean_i = max(1.0, np.sum(blend[:, :, i] * mask_i) / (mask_i.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        h, w = r_blur.shape[:2]
        new_w = np.random.randint(0, max_image_size - w - 10) if w < max_image_size - 10 else 0
        new_h = np.random.randint(0, max_image_size - h - 10) if h < max_image_size - 10 else 0

        g_mask = _gen_refool_kernel(max_image_size, 3)
        g_mask = np.dstack((g_mask, g_mask, g_mask))
        alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1.0 - alpha_t / 2.0)

        r_blur_mask = np.multiply(r_blur, alpha_r)
        blur_r = min(1.0, 4 * (1 - alpha_t)) * r_blur_mask
        blend = r_blur_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1.0 / 2.2)
        r_blur_mask = np.power(blur_r, 1.0 / 2.2)
        blend = np.power(blend, 1.0 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0

        blended = np.uint8(blend * 255)
        reflection_layer = np.uint8(r_blur_mask * 255)
        transmission_layer = np.uint8(transmission_layer * 255)

    return blended, transmission_layer, reflection_layer


def _gen_refool_kernel(kern_len=100, nsig=1):
    interval = (2 * nsig + 1.0) / kern_len
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kern_len + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel / kernel.max()


def apply_sig_trigger(img, args):
    params = _build_trigger_params(args, "sig")
    arr = np.asarray(_as_pil(img).convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    amplitude = float(params.sig_delta)
    if amplitude > 1.0:
        amplitude /= 255.0
    frequency = float(params.sig_frequency)
    direction = str(params.sig_direction).lower()

    length = h if direction == "vertical" else w
    axis = np.arange(length, dtype=np.float32)
    signal = amplitude * np.sin(2 * np.pi * frequency * axis / length)
    if direction == "vertical":
        arr += signal[:, None, None]
    else:
        arr += signal[None, :, None]

    return Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255), mode="RGB")


def apply_wanet_trigger(img, args):
    params = _build_trigger_params(args, "wanet")
    img_pil = _as_pil(img).convert("RGB")
    arr = np.asarray(img_pil, dtype=np.uint8)
    h, w = arr.shape[:2]
    map_x, map_y = _get_wanet_maps(args, h, w, params)
    warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(warped, mode="RGB")


def _get_wanet_maps(args, h, w, params):
    k = int(params.wanet_k)
    strength = float(params.wanet_strength)
    seed = params.wanet_seed
    key = (h, w, k, strength, seed)

    cache = _get_arg(args, "_wanet_map_cache", None)
    if cache is None:
        cache = {}
        _set_arg(args, "_wanet_map_cache", cache)
    if key in cache:
        return cache[key]

    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1.0, 1.0, size=(k, k, 2)).astype(np.float32)
    dx = cv2.resize(noise[:, :, 0], (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(noise[:, :, 1], (w, h), interpolation=cv2.INTER_CUBIC)
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=max(1.0, min(h, w) / (2.0 * k)))
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=max(1.0, min(h, w) / (2.0 * k)))

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(grid_x + dx * strength, 0, w - 1).astype(np.float32)
    map_y = np.clip(grid_y + dy * strength, 0, h - 1).astype(np.float32)
    cache[key] = (map_x, map_y)
    return cache[key]
