"""End-to-end BDetCLIP evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_backdoor.datasets.pre_resize import resolve_pre_resize
from ssl_backdoor.defenses.subspace_detection.data import ImageSampleDataset, read_samples
from ssl_backdoor.defenses.subspace_detection.modeling import (
    HuggingFaceVisionLanguageEncoder,
    normalize,
)
from ssl_backdoor.utils.utils import set_seed

from .data import MixedTriggeredDataset, split_samples
from .prompts import PromptBank


def aggregate(features: torch.Tensor, groups: int, items_per_group: int) -> torch.Tensor:
    if len(features) != groups * items_per_group:
        raise ValueError("text feature count does not match the requested groups")
    return normalize(features.view(groups, items_per_group, -1).mean(1))


def build_direction(benign: torch.Tensor, malignant: torch.Tensor) -> torch.Tensor:
    if benign.shape != malignant.shape:
        raise ValueError("benign and malignant feature shapes must match")
    return (benign - malignant).sum(0)


def score_features(features: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return normalize(features) @ direction


def detection_metrics(omega, poisoned, reference_omega):
    omega = np.asarray(omega, dtype=np.float64)
    poisoned = np.asarray(poisoned, dtype=bool)
    reference_omega = np.asarray(reference_omega, dtype=np.float64)
    if not poisoned.any() or poisoned.all() or not len(reference_omega):
        raise ValueError(
            "metrics require reference, clean evaluation, and poisoned evaluation samples"
        )
    threshold = float(reference_omega.min())
    anomaly = -omega
    detected = omega < threshold
    return {
        "auroc": float(roc_auc_score(poisoned, anomaly)),
        "auprc": float(average_precision_score(poisoned, anomaly)),
        "threshold": threshold,
        "threshold_accuracy": float(accuracy_score(poisoned, detected)),
        "threshold_precision": float(precision_score(poisoned, detected, zero_division=0)),
        "threshold_recall": float(recall_score(poisoned, detected, zero_division=0)),
        "threshold_f1": float(f1_score(poisoned, detected, zero_division=0)),
        "clean_omega_mean": float(omega[~poisoned].mean()),
        "poisoned_omega_mean": float(omega[poisoned].mean()),
    }


def _loader(dataset, runtime):
    return DataLoader(
        dataset,
        batch_size=int(runtime.get("batch_size", 64)),
        num_workers=int(runtime.get("workers", 4)),
        pin_memory=True,
        shuffle=False,
    )


@torch.inference_mode()
def _encode_reference(loader, encoder):
    features = []
    for pixels, _, _ in tqdm(loader, desc="Encoding reference images"):
        features.append(encoder.encode_images(pixels).cpu())
    return torch.cat(features)


@torch.inference_mode()
def _encode_evaluation(loader, encoder):
    features, labels, paths, poisoned = [], [], [], []
    for pixels, batch_labels, batch_paths, batch_poisoned in tqdm(
        loader, desc="Encoding evaluation images"
    ):
        features.append(encoder.encode_images(pixels).cpu())
        labels.append(batch_labels)
        paths.extend(batch_paths)
        poisoned.append(batch_poisoned)
    return torch.cat(features), torch.cat(labels), paths, torch.cat(poisoned).bool()


def _text_features(encoder, bank, text_batch_size, prediction_template):
    benign = aggregate(encoder.encode_texts(bank.benign_texts(), text_batch_size), 1000, 7)
    malignant = aggregate(
        encoder.encode_texts(bank.malignant_texts(), text_batch_size), 1000, len(bank.templates)
    )
    prototypes = encoder.encode_texts(
        [prediction_template.format(name) for name in bank.classes], text_batch_size
    )
    return build_direction(benign, malignant), prototypes


def _target_index(target, classes):
    if isinstance(target, int):
        if 0 <= target < len(classes):
            return target
        raise ValueError("target_class is out of range")
    try:
        return classes.index(target)
    except ValueError as error:
        raise ValueError(f"unknown target_class: {target}") from error


def run_bdetclip(config: dict, model=None, processor=None) -> dict:
    pre_resize, pre_resize_size = resolve_pre_resize(config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    resources = config["resources"]
    bank = PromptBank.load(
        resources["classes"], resources["benign_prompts"], resources["malignant_prompts"]
    )
    target = _target_index(config.get("target_class", "banana"), bank.classes)
    encoder = (
        HuggingFaceVisionLanguageEncoder.from_config(config["model"], device)
        if model is None
        else HuggingFaceVisionLanguageEncoder(model, processor, device)
    )

    data, runtime = config["data"], config.get("runtime", {})
    samples = read_samples(data["labels_csv"], data.get("image_root"))
    reference, evaluation, poisoned_indices = split_samples(
        samples,
        reference_samples=int(data.get("reference_samples", 200)),
        evaluation_samples=int(data.get("evaluation_samples", 1000)),
        poison_ratio=float(data.get("poison_ratio", 0.3)),
        target=target,
        seed=seed,
    )
    resize_args = {"pre_resize": pre_resize, "pre_resize_size": pre_resize_size}
    reference_features = _encode_reference(
        _loader(
            ImageSampleDataset(reference, encoder.process_image, **resize_args),
            runtime,
        ),
        encoder,
    )
    features, labels, paths, poisoned = _encode_evaluation(
        _loader(
            MixedTriggeredDataset(
                evaluation,
                poisoned_indices,
                encoder.process_image,
                config["trigger"],
                seed,
                **resize_args,
            ),
            runtime,
        ),
        encoder,
    )
    direction, prototypes = _text_features(
        encoder,
        bank,
        int(runtime.get("text_batch_size", 256)),
        config.get("prediction_template", "a photo of a {}"),
    )
    reference_omega = score_features(reference_features, direction).numpy()
    omega = score_features(features, direction).numpy()
    reference_predictions = (normalize(reference_features) @ prototypes.T).argmax(1)
    predictions = (normalize(features) @ prototypes.T).argmax(1)
    result = detection_metrics(omega, poisoned.numpy(), reference_omega)
    non_target = labels != target
    poisoned_non_target = poisoned & non_target
    result.update(
        {
            "experiment_id": config.get("experiment_id", "bdetclip"),
            "model_name": config["model"]["name"],
            "checkpoint": config["model"].get("checkpoint"),
            "target_index": target,
            "num_reference": len(reference),
            "num_evaluation": len(evaluation),
            "num_clean": int((~poisoned).sum()),
            "num_poisoned": int(poisoned.sum()),
            "clean_accuracy": float((predictions[~poisoned] == labels[~poisoned]).float().mean()),
            "asr": float((predictions[poisoned_non_target] == target).float().mean()),
        }
    )

    output_dir = Path(config["output_dir"]) / result["experiment_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    detected = omega < result["threshold"]
    with (output_dir / "scores.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "split",
            "path",
            "label",
            "predicted",
            "is_poisoned",
            "omega",
            "anomaly_score",
            "detected",
        ])
        for (path, label), predicted, raw_score in zip(
            reference, reference_predictions.tolist(), reference_omega
        ):
            writer.writerow(
                ["reference", path, label, predicted, 0, float(raw_score), -float(raw_score), 0]
            )
        rows = zip(
            paths,
            labels.tolist(),
            predictions.tolist(),
            poisoned.tolist(),
            omega,
            detected,
        )
        for row in rows:
            path, label, predicted, is_poisoned, raw_score, is_detected = row
            writer.writerow([
                "evaluation",
                path,
                label,
                predicted,
                int(is_poisoned),
                float(raw_score),
                -float(raw_score),
                int(is_detected),
            ])
    return result
