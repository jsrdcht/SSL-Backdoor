import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

from ssl_backdoor.evaluation import (
    CLIPZeroShotEvaluator,
    build_clip_text_prototypes,
)


class TinyTokenizer:
    def __call__(self, *, text, **kwargs):
        input_ids = torch.tensor([0 if "cat" in prompt else 1 for prompt in text])
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("basis", torch.eye(2))

    def get_text_features(self, input_ids, attention_mask=None):
        return self.basis[input_ids]

    def get_image_features(self, pixel_values):
        return pixel_values


class TinyOpenCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("basis", torch.eye(2))

    def encode_text(self, tokens):
        return self.basis[tokens]

    def encode_image(self, images):
        return images


class TinyOpenCLIPTokenizer:
    def __call__(self, prompts):
        return torch.tensor([0 if "cat" in prompt else 1 for prompt in prompts])


class MappingDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "pixel_values": self.images[index],
            "labels": self.labels[index],
        }


def test_build_text_prototypes_accepts_callable_and_format_templates():
    prototypes = build_clip_text_prototypes(
        TinyCLIP(),
        TinyTokenizer(),
        ["cat", "dog"],
        [lambda name: f"a photo of a {name}", "a sketch of a {}"],
        "cpu",
    )
    assert torch.equal(prototypes, torch.eye(2))


def test_evaluator_uses_supplied_tuple_dataset_and_reports_topk_accuracy():
    dataset = TensorDataset(
        torch.tensor([[1.0, 0.0], [0.1, 0.9], [0.8, 0.2]]),
        torch.tensor([0, 1, 1]),
    )
    evaluator = CLIPZeroShotEvaluator(
        TinyCLIP(),
        TinyTokenizer(),
        ["cat", "dog"],
        ["a photo of a {}"],
        device="cpu",
    )
    result = evaluator.evaluate(dataset, topk=(1, 2), batch_size=2, num_workers=0)

    assert result.predictions.tolist() == [0, 1, 0]
    assert result.topk_accuracy(1) == torch.tensor(2 / 3).item()
    assert result.topk_accuracy(2) == 1.0


def test_evaluator_accepts_mapping_dataset():
    dataset = MappingDataset(torch.eye(2), torch.tensor([0, 1]))
    evaluator = CLIPZeroShotEvaluator(
        TinyCLIP(), TinyTokenizer(), ["cat", "dog"], [lambda name: name]
    )

    result = evaluator.evaluate(dataset, batch_size=2, num_workers=0)

    assert result.topk_accuracy() == 1.0


def test_evaluator_accepts_openclip_style_model_and_tokenizer():
    evaluator = CLIPZeroShotEvaluator(
        TinyOpenCLIP(),
        TinyOpenCLIPTokenizer(),
        ["cat", "dog"],
        ["a photo of a {}"],
    )
    result = evaluator.evaluate(
        TensorDataset(torch.eye(2), torch.tensor([0, 1])),
        batch_size=2,
        num_workers=0,
    )

    assert result.topk_accuracy() == 1.0
