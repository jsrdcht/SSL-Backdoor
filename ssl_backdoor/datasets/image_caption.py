"""Image-caption dataset for CLIP image-text contrastive training."""
import csv
import os

from PIL import Image
from torch.utils.data import Dataset


class ImageCaptionDataset(Dataset):
    """Read (image, caption) pairs from CSV/TSV, preprocessing delegated to HuggingFace CLIPProcessor."""

    def __init__(self, csv_path, processor, image_key='image', caption_key='caption',
                 delimiter=',', image_root=None, max_length=None):
        self.processor = processor
        self.image_root = image_root or os.path.dirname(os.path.abspath(csv_path))
        if max_length is None and hasattr(processor, 'tokenizer'):
            max_length = getattr(processor.tokenizer, 'model_max_length', None)
            if not isinstance(max_length, int) or max_length > 10000:
                max_length = 77
        self.max_length = max_length

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            missing = {image_key, caption_key} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV {csv_path} missing columns {sorted(missing)}, available columns: {reader.fieldnames}")
            self.samples = [(row[image_key], row[caption_key]) for row in reader]
        if not self.samples:
            raise ValueError(f"CSV {csv_path} contains no samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, caption = self.samples[index]
        image_path = path if os.path.isabs(path) else os.path.join(self.image_root, path)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {image_path} ({e})") from e
        encoded = self.processor(images=image, text=caption, padding='max_length',
                                 truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'pixel_values': encoded['pixel_values'][0],
            'input_ids': encoded['input_ids'][0],
            'attention_mask': encoded['attention_mask'][0],
            'image_path': image_path,
            'caption': caption,
        }
