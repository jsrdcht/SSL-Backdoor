"""ImageNet CSV data adapters for Subspace Detection."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ssl_backdoor.datasets.attacker.triggers import apply_static_trigger
from ssl_backdoor.datasets.attacker.trigger_templates import validate_trigger_config
from ssl_backdoor.datasets.pre_resize import pre_resize_image


class ImagePathResolver:
    def __init__(self, image_root: str | None):
        self.root = Path(image_root).expanduser() if image_root else None
        self.by_name = None

    def __call__(self, raw_path: str, csv_path: Path) -> Path:
        path = Path(raw_path).expanduser()
        candidates = [path]
        if not path.is_absolute():
            relative = [self.root / path if self.root else None, csv_path.parent / path]
            candidates.extend(filter(None, relative))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        if self.root is None:
            raise FileNotFoundError(f"image not found: {raw_path}")
        if self.by_name is None:
            self.by_name = {item.name: item for item in self.root.rglob("*") if item.is_file()}
        resolved = self.by_name.get(path.name)
        if resolved is None:
            raise FileNotFoundError(f"image {path.name} not found under {self.root}")
        return resolved


def read_samples(labels_csv: str, image_root: str | None = None) -> list[tuple[Path, int]]:
    csv_path = Path(labels_csv).expanduser()
    resolver = ImagePathResolver(image_root)
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = {"image", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        rows = [(resolver(row["image"], csv_path), int(row["label"])) for row in reader]
    if not rows:
        raise ValueError(f"{csv_path} contains no samples")
    return rows


class ImageSampleDataset(Dataset):
    def __init__(self, samples, process_image, pre_resize=False, pre_resize_size=None):
        self.samples = list(samples)
        self.process_image = process_image
        self.pre_resize = pre_resize
        self.pre_resize_size = pre_resize_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = pre_resize_image(image, self.pre_resize, self.pre_resize_size)
        return self.process_image(image), label, str(path)


class PairedTriggeredDataset(Dataset):
    def __init__(
        self,
        samples,
        process_image,
        trigger: dict,
        seed: int,
        seed_offset: int = 0,
        pre_resize=False,
        pre_resize_size=None,
    ):
        self.samples = list(samples)
        self.process_image = process_image
        self.trigger = dict(trigger)
        validate_trigger_config(self.trigger, require_path=True)
        self.seed = seed
        self.seed_offset = seed_offset
        self.pre_resize = pre_resize
        self.pre_resize_size = pre_resize_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = pre_resize_image(image, self.pre_resize, self.pre_resize_size)
        clean = self.process_image(image)
        item_seed = self.seed + self.seed_offset + index
        random.seed(item_seed)
        np.random.seed(item_seed % (2**32))
        torch.manual_seed(item_seed)
        poisoned = apply_static_trigger(
            image, self.trigger, trigger=self.trigger.get("trigger_path")
        )
        return clean, self.process_image(poisoned), label, str(path)
