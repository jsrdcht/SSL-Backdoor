"""Load paper resources and generate style, translation, and description variants."""

from __future__ import annotations

import csv
from pathlib import Path


RESOURCE_DIR = Path(__file__).resolve().parents[3] / "assets/subspace_detection_resources"
DEFAULT_DESCRIPTIONS = RESOURCE_DIR / "imagenet1k_descriptions.csv"
DEFAULT_ARABIC_CLASSES = RESOURCE_DIR / "imagenet1k_arabic_classes.txt"
_DESCRIPTION_FIELDS = [f"description_{i}" for i in range(1, 7)]
_BOLD_LOWER = "".join(chr(0x1D5EE + i) for i in range(26))
_BOLD_UPPER = "".join(chr(0x1D5D4 + i) for i in range(26))
_ITALIC_LOWER = "".join("ℎ" if i == 7 else chr(0x1D44E + i) for i in range(26))
_ITALIC_UPPER = "".join(chr(0x1D434 + i) for i in range(26))


def _styled(text: str, lower: str, upper: str) -> str:
    return text.translate(
        str.maketrans("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", lower + upper)
    )


class TextVariantBank:
    """Provide the nine paper-specified text variants for ImageNet-1K classes."""

    def __init__(
        self,
        descriptions_path: str | Path = DEFAULT_DESCRIPTIONS,
        arabic_classes_path: str | Path = DEFAULT_ARABIC_CLASSES,
    ):
        with Path(descriptions_path).open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        indices = [int(row["class_index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError("description class_index values must be contiguous from zero")
        self.classes = [row["class_name"].strip() for row in rows]
        self.descriptions = [
            [row[field].strip() for field in _DESCRIPTION_FIELDS] for row in rows
        ]
        self.sources = [row.get("source", "").strip() for row in rows]
        self.source_notes = [row.get("source_note", "").strip() for row in rows]
        arabic_text = Path(arabic_classes_path).read_text(encoding="utf-8")
        self.arabic = [line.strip() for line in arabic_text.splitlines()]
        if not self.classes or len(self.arabic) != len(self.classes):
            raise ValueError("English classes, Arabic classes, and descriptions are misaligned")
        if any(not name for name in self.classes + self.arabic):
            raise ValueError("text resources contain an empty class name")
        if any(len(items) != 6 or any(not item for item in items) for items in self.descriptions):
            raise ValueError("each class must contain six non-empty descriptions")

    def __len__(self) -> int:
        return len(self.classes)

    def variants(self, class_index: int) -> list[str]:
        label = self.classes[class_index]
        return [
            _styled(label, _BOLD_LOWER, _BOLD_UPPER),
            _styled(label, _ITALIC_LOWER, _ITALIC_UPPER),
            self.arabic[class_index],
            *self.descriptions[class_index],
        ]

    def texts(self, class_index: int) -> list[str]:
        """Return the original class name followed by its nine variants."""
        return [self.classes[class_index], *self.variants(class_index)]
