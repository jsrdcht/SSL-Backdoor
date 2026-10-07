import logging

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.defenses import patchsearch
from ssl_backdoor.defenses.patchsearch import poison_classifier
from ssl_backdoor.defenses.patchsearch.utils.dataset import denormalize, get_transforms
from ssl_backdoor.defenses.patchsearch.utils.model_utils import get_model


@pytest.mark.parametrize("labels, expected", [([0, 0, 1, 1], 1.0), ([0, 0, 0, 0], 0.0)])
def test_search_reports_ranking_metrics(tmp_path, monkeypatch, labels, expected):
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    order = np.argsort(-scores)
    monkeypatch.setattr(
        patchsearch, "patchsearch_iterative",
        lambda **kwargs: (scores, order, np.array(labels)),
    )
    dataset = TensorDataset(torch.zeros(4, 3, 4, 4), torch.zeros(4))
    results = patchsearch.run_patchsearch(
        {}, model=torch.nn.Identity(), suspicious_dataset=dataset,
        output_dir=str(tmp_path), num_workers=0, topk_thresholds=[2],
    )

    assert results["auroc"] == expected
    assert results["auprc"] == expected
    np.testing.assert_array_equal(np.load(tmp_path / "defense_run/sorted_indices.npy"), order)


def test_filter_uses_probabilities_and_global_sample_indices(caplog):
    # All predictions are clean, but the probability ranking is perfect.
    probabilities = torch.tensor([0.1, 0.4, 0.2, 0.3])
    logits = torch.stack((1 - probabilities, probabilities), dim=1).log()
    indices = torch.tensor([2, 0, 3, 1])
    labels = torch.tensor([0, 1, 0, 1])
    dataset = [("", logits[i], 0, labels[i], indices[i]) for i in range(4)]
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))

    with caplog.at_level(logging.INFO, logger="patchsearch"):
        recall, precision, predictions = poison_classifier.test(
            DataLoader(dataset, batch_size=2), model, None,
        )

    assert recall == precision == 0.0
    np.testing.assert_array_equal(predictions, np.zeros(4))
    assert "AUROC: 100.00%" in caplog.text
    assert "AUPRC (Average Precision): 100.00%" in caplog.text


def test_cifar100_normalization_matches_dataset_registry():
    transform = get_transforms("cifar100", 32)
    assert transform.transforms[-1] is dataset_params["cifar100"]["normalize"]
    image = Image.new("RGB", (32, 32), (60, 120, 180))
    restored = denormalize(transform(image), "cifar100")
    expected = torch.tensor([60, 120, 180]).float() / 255
    torch.testing.assert_close(restored, expected.expand(32, 32, 3))


@pytest.mark.parametrize("prefix, dataset, small_conv", [
    ("", "cifar10", True),
    ("module.encoder_q.", "cifar10", True),
    ("module.base_encoder.", "cifar10", True),
    ("module.encoder_q.", "stl10", True),
    ("", "stl10", False),
])
def test_loads_legacy_encoder_weights_without_silent_partial_loading(
    tmp_path, prefix, dataset, small_conv,
):
    encoder = resnet18()
    if small_conv:
        encoder.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    if dataset == "cifar10":
        encoder.maxpool = torch.nn.Identity()
    encoder.fc = torch.nn.Identity()
    checkpoint = tmp_path / "encoder.pth"
    state = {prefix + key: value for key, value in encoder.state_dict().items()}
    # Projection heads must not become encoder parameters.
    state[prefix + "fc.weight"] = torch.randn(2, 2)
    torch.save({"state_dict": state}, checkpoint)

    loaded = get_model("moco_resnet18", str(checkpoint), dataset)

    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert type(loaded.maxpool) is type(encoder.maxpool)
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
