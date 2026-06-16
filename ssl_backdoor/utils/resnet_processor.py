from typing import List, Union

import torch
from PIL import Image
import torchvision.transforms as transforms


class _ProcessorOutput(dict):
    """简单字典，使其支持 .to(device) 以与 HuggingFace 的 BatchEncoding 接口保持一致。"""

    def to(self, device: torch.device):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)
        return self


class ResNetImageProcessor:
    """为 ResNet-18 提供与 `AutoImageProcessor` 类似的接口，方便统一调用。

    主要完成以下任务：
    1. 图像 resize + center crop（保持 224×224 输入尺寸）。
    2. 转为张量并执行 ImageNet 归一化（mean, std）。
    3. 返回字典类型结果，并实现 `.to(device)` 方法便于转移到 GPU。

    使用示例：
    >>> processor = ResNetImageProcessor()  # 或 ResNetImageProcessor(size=224)
    >>> inputs = processor(images=pil_img, return_tensors="pt")
    >>> inputs = inputs.to("cuda")
    >>> feats = model(inputs)  # 与 DINOv2 的调用保持一致
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
    def from_pretrained(cls, *args, **kwargs):  # 保留兼容接口，无需权重
        """保持与 `AutoImageProcessor.from_pretrained` 接口一致。"""
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