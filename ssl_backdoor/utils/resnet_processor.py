from typing import List, Union

import torch
from PIL import Image
import torchvision.transforms as transforms


class _ProcessorOutput(dict):
    """."""

    def to(self, device: torch.device):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)
        return self


class ResNetImageProcessor:
    """

        
        
        
        

        
        
    >>> inputs = processor(images=pil_img, return_tensors="pt")
    >>> inputs = inputs.to("cuda")
        
    """

    image_mean: List[float] = [0.485, 0.456, 0.406]
    image_std: List[float] = [0.229, 0.224, 0.225]

    def __init__(self, size: int = 224):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std),
        ])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """."""
        return cls(**kwargs)

    def __call__(self, *, images: Union[Image.Image, List[Image.Image]], return_tensors: str = "pt") -> _ProcessorOutput:
        if not isinstance(images, (list, tuple)):
            images = [images]
        processed: List[torch.Tensor] = [self.transform(img.convert("RGB")) for img in images]
        pixel_values = torch.stack(processed, dim=0)
        if return_tensors == "pt":
            return _ProcessorOutput({"pixel_values": pixel_values})
        else:
            raise ValueError(f"Unsupported return_tensors value: {return_tensors}") 