import argparse

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
