"""CLIP backdoor shared utilities: CSV read/write, image root directory resolution, trigger args assembly."""
import csv
import os

from ssl_backdoor.datasets.attacker.trigger_templates import validate_trigger_config


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


def build_trigger_args(trigger_cfg):
    """Validate a dedicated trigger section and preserve its canonical fields."""
    validate_trigger_config(trigger_cfg, require_path=True)
    return dict(trigger_cfg)
