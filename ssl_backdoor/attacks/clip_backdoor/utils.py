"""CLIP backdoor shared utilities: CSV read/write, image root directory resolution, trigger args assembly."""
import csv
import os


def read_image_caption_csv(csv_path, image_key="image", caption_key="caption", delimiter=","):
    """Read (image, caption) CSV, return (rows, fieldnames). Keep other columns for traceability."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        missing = {image_key, caption_key} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV {csv_path} missing columns {sorted(missing)}, available columns: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV {csv_path} contains no samples")
    return rows, list(reader.fieldnames)


def write_csv(path, rows, fieldnames, delimiter=","):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_root(csv_path, image_root=None):
    """Consistent with ImageCaptionDataset: relative paths based on CSV directory, can override with image_root."""
    return image_root or os.path.dirname(os.path.abspath(csv_path))


def resolve_image_path(path, image_root):
    return path if os.path.isabs(path) else os.path.join(image_root, path)


# Trigger fields passed to apply_static_trigger (other algorithm-specific fields copied directly from trigger section).
_TRIGGER_KEYS = (
    "attack_algorithm", "trigger_insert", "trigger_path", "reflection_path",
    "trigger_size", "position", "location_min", "location_max", "alpha", "alpha_t",
    "attack_magnitude", "channel_list", "window_size", "pos_list", "lindct",
    "sig_delta", "sig_frequency", "sig_direction", "ghost_rate", "offset", "sigma",
    "ghost_alpha", "wanet_k", "wanet_strength", "wanet_seed", "generator_path", "device",
)


def build_trigger_args(trigger_cfg):
    """Build trigger arguments while preserving the configured selector."""
    args = {k: trigger_cfg[k] for k in _TRIGGER_KEYS if k in trigger_cfg}
    if not (trigger_cfg.get("attack_algorithm") or trigger_cfg.get("trigger_insert")):
        raise ValueError("trigger config must provide attack_algorithm or trigger_insert")
    return args
