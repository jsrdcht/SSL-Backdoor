import argparse

import torch
import yaml

from tools import run_decomp


def _write_yaml(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_reference_and_validation_datasets_share_pre_resize_args(
    tmp_path, monkeypatch
):
    reference_file = tmp_path / "images.txt"
    reference_file.write_text(
        "/unused/a.png 0\n/unused/b.png 1\n", encoding="utf-8"
    )
    config_path = tmp_path / "decomp.yaml"
    _write_yaml(
        config_path,
        {
            "reference_file": str(reference_file),
            "model_path": "/unused/model",
            "dataset_name": "cifar10",
            "device": "cpu",
            "clean_ratio": 0.5,
            "output_dir": str(tmp_path / "output"),
        },
    )
    poison_config_path = tmp_path / "poison.yaml"
    _write_yaml(
        poison_config_path,
        {
            "dataset": "cifar10",
            "attack_algorithm": "clean",
            "pre_resize": True,
            "pre_resize_size": 6,
        },
    )

    dataset_args = []

    def capture_dataset(args, *_args, **_kwargs):
        dataset_args.append(args)
        return object()

    monkeypatch.setattr(
        run_decomp,
        "get_args",
        lambda: argparse.Namespace(
            config=str(config_path), poison_config=str(poison_config_path)
        ),
    )
    monkeypatch.setattr(run_decomp, "FileListDataset", capture_dataset)
    monkeypatch.setattr(
        run_decomp, "OnlineUniversalPoisonedValDataset", capture_dataset
    )
    monkeypatch.setattr(run_decomp, "DataLoader", lambda dataset, **_kwargs: dataset)
    stub_results = {
        "load_model": (object(), None),
        "extract_prs_features": (None, None),
        "get_classes_and_templates": (["class"], ["{}"]),
        "get_zero_shot_classifier": None,
        "run_image_detection_and_ablation": ({}, [], []),
        "calculate_detection_metrics": {},
    }
    for name, result in stub_results.items():
        monkeypatch.setattr(
            run_decomp,
            name,
            lambda *args, _result=result, **kwargs: _result,
        )

    run_decomp.main()

    assert len(dataset_args) == 3
    assert {(args.pre_resize, args.pre_resize_size) for args in dataset_args} == {
        (True, 6)
    }


def test_zero_shot_classifier_preserves_prompt_averaging_and_processor_length():
    prompt_ids = {"photo cat": 0, "sketch cat": 1, "photo dog": 2, "sketch dog": 3}
    token_lengths = []

    def processor(*, text, padding, truncation, return_tensors, max_length=5):
        assert (padding, truncation, return_tensors) == ("max_length", True, "pt")
        token_lengths.append(max_length)
        input_ids = [[prompt_ids[prompt]] * max_length for prompt in text]
        return {"input_ids": torch.tensor(input_ids)}

    class TinyCLIP(torch.nn.Module):
        def get_text_features(self, input_ids):
            # Different norms expose averaging before per-prompt normalization.
            features = torch.tensor(
                [[3., 0., 0.], [0., 4., 0.], [0., 0., 2.], [0., 6., 0.]]
            )
            return features[input_ids[:, 0]]

    prototypes = run_decomp.get_zero_shot_classifier(
        TinyCLIP(), ["cat", "dog"], "cpu", processor, ["photo {}", "sketch {}"]
    )

    expected = torch.tensor([[1., 0.], [1., 1.], [0., 1.]]) / 2**0.5
    torch.testing.assert_close(prototypes, expected)
    assert token_lengths == [5, 5]
