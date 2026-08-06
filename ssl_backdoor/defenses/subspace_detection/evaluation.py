"""End-to-end Subspace Detection evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_backdoor.utils.utils import set_seed

from .data import ImageSampleDataset, PairedTriggeredDataset, read_samples
from .detector import SubspaceDetector
from .modeling import HuggingFaceVisionLanguageEncoder, normalize
from .text_variants import TextVariantBank


def _loader(dataset, runtime):
    return DataLoader(
        dataset,
        batch_size=int(runtime.get("batch_size", 64)),
        num_workers=int(runtime.get("workers", 4)),
        pin_memory=True,
        shuffle=False,
    )


@torch.no_grad()
def _encode(loader, encoder, paired=False):
    clean, poisoned, labels, paths = [], [], [], []
    for batch in tqdm(loader, desc="Encoding images"):
        if paired:
            clean_pixels, poisoned_pixels, batch_labels, batch_paths = batch
            poisoned.append(encoder.encode_images(poisoned_pixels).cpu())
        else:
            clean_pixels, batch_labels, batch_paths = batch
        clean.append(encoder.encode_images(clean_pixels).cpu())
        labels.append(batch_labels)
        paths.extend(batch_paths)
    values = (torch.cat(clean), torch.cat(labels), paths)
    return (*values, torch.cat(poisoned)) if paired else values


def _predict(features, prototypes):
    return (normalize(features) @ normalize(prototypes).T).argmax(1)


def _fit_distributions(encoder, bank, detector, class_indices, text_batch_size):
    ordered = sorted(set(map(int, class_indices)))
    texts = [text for index in ordered for text in bank.texts(index)]
    features = encoder.encode_texts(texts, text_batch_size).view(len(ordered), 10, -1)
    return {
        index: detector.fit_class(class_features).cpu()
        for index, class_features in tqdm(
            zip(ordered, features), total=len(ordered), desc="Fitting text subspaces"
        )
    }


def _score(features, predicted, distributions):
    scores = torch.empty(len(features))
    for class_index in predicted.unique().tolist():
        mask = predicted == class_index
        scores[mask] = SubspaceDetector.score(features[mask], distributions[class_index])
    return scores.numpy()


def _metrics(clean_scores, poisoned_scores, reference_scores):
    labels = np.r_[np.zeros(len(clean_scores)), np.ones(len(poisoned_scores))]
    scores = np.r_[clean_scores, poisoned_scores]
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    reference_threshold = float(np.max(reference_scores))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "best_f1": float(f1[best_index]),
        "best_f1_threshold": float(thresholds[best_index]) if len(thresholds) else None,
        "reference_threshold": reference_threshold,
        "reference_threshold_f1": float(f1_score(labels, scores > reference_threshold)),
        "clean_score_mean": float(np.mean(clean_scores)),
        "poisoned_score_mean": float(np.mean(poisoned_scores)),
    }


def _resolve_target(target, classes):
    if isinstance(target, int):
        if not 0 <= target < len(classes):
            raise ValueError("target_class is out of range")
        return target
    try:
        return classes.index(target)
    except ValueError as error:
        raise ValueError(f"unknown target_class: {target}") from error


def run_subspace_detection(config: dict, model=None, processor=None) -> dict:
    """Run the complete detection evaluation and return its metrics."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    resources = config.get("resources", {})
    missing_resources = {"descriptions", "arabic_classes"} - resources.keys()
    if missing_resources:
        raise ValueError(f"resources is missing required keys: {sorted(missing_resources)}")
    bank = TextVariantBank(resources["descriptions"], resources["arabic_classes"])
    encoder = (
        HuggingFaceVisionLanguageEncoder.from_config(config["model"], device)
        if model is None
        else HuggingFaceVisionLanguageEncoder(model, processor, device)
    )
    detector = SubspaceDetector(seed=seed, **config.get("detector", {}))
    data, runtime = config["data"], config.get("runtime", {})
    samples = read_samples(data["labels_csv"], data.get("image_root"))
    start = int(data.get("start_index", 0))
    requested_samples = data.get("num_samples")

    reference_csv = data.get("reference_csv")
    if reference_csv:
        reference_root = data.get("reference_image_root", data.get("image_root"))
        reference_pool = read_samples(reference_csv, reference_root)
        num_reference = int(data.get("reference_samples", len(reference_pool)))
        reference_samples = reference_pool[:num_reference]
        available = len(samples) - start
        num_samples = int(requested_samples if requested_samples is not None else available)
        clean_samples = samples[start : start + num_samples]
        clean_start = start
    else:
        num_reference = int(data.get("reference_samples", 200))
        reference_samples = samples[start : start + num_reference]
        available = len(samples) - start - num_reference
        num_samples = int(requested_samples if requested_samples is not None else available)
        clean_samples = samples[start + num_reference : start + num_reference + num_samples]
        clean_start = start + num_reference
    if not reference_samples:
        raise ValueError("the clean reference set must not be empty")
    if len(reference_samples) != int(data.get("reference_samples", len(reference_samples))):
        raise ValueError("not enough clean reference samples")
    if num_samples <= 0 or len(clean_samples) != num_samples:
        raise ValueError("not enough clean evaluation samples")

    trigger = config.get("trigger")
    poisoned_csv = data.get("poisoned_csv")
    if poisoned_csv:
        poisoned_root = data.get("poisoned_image_root", data.get("image_root"))
        poisoned_pool = read_samples(poisoned_csv, poisoned_root)
        poisoned_samples = poisoned_pool[:num_samples]
        if len(poisoned_samples) != num_samples:
            raise ValueError("not enough poisoned evaluation samples")
        clean_features, labels, paths = _encode(
            _loader(ImageSampleDataset(clean_samples, encoder.process_image), runtime), encoder
        )
        poisoned_features, poisoned_labels, poisoned_paths = _encode(
            _loader(ImageSampleDataset(poisoned_samples, encoder.process_image), runtime), encoder
        )
    else:
        if not trigger:
            raise ValueError("trigger is required when poisoned_csv is not provided")
        clean_features, labels, paths, poisoned_features = _encode(
            _loader(
                PairedTriggeredDataset(
                    clean_samples, encoder.process_image, trigger, seed, seed_offset=clean_start
                ),
                runtime,
            ),
            encoder,
            paired=True,
        )
        poisoned_labels, poisoned_paths = labels, paths

    reference_features, _, _ = _encode(
        _loader(ImageSampleDataset(reference_samples, encoder.process_image), runtime), encoder
    )
    template = config.get("prediction_template", "a photo of a {}")
    prototypes = encoder.encode_texts(
        [template.format(name) for name in bank.classes], int(runtime.get("text_batch_size", 256))
    )
    reference_pred = _predict(reference_features, prototypes)
    clean_pred = _predict(clean_features, prototypes)
    poisoned_pred = _predict(poisoned_features, prototypes)
    distributions = _fit_distributions(
        encoder,
        bank,
        detector,
        torch.cat([reference_pred, clean_pred, poisoned_pred]).tolist(),
        int(runtime.get("text_batch_size", 256)),
    )
    reference_scores = _score(reference_features, reference_pred, distributions)
    clean_scores = _score(clean_features, clean_pred, distributions)
    poisoned_scores = _score(poisoned_features, poisoned_pred, distributions)
    result = _metrics(clean_scores, poisoned_scores, reference_scores)
    result.update(
        {
            "experiment_id": config.get("experiment_id", "subspace_detection"),
            "model_name": config["model"]["name"],
            "checkpoint": config["model"].get("checkpoint"),
            "num_reference": len(reference_samples),
            "num_clean": len(clean_samples),
            "num_poisoned": len(poisoned_features),
            "clean_accuracy": float((clean_pred == labels).float().mean()),
        }
    )
    if "target_class" in config:
        target = _resolve_target(config["target_class"], bank.classes)
        hits = poisoned_pred == target
        non_target = poisoned_labels != target
        result["target_index"] = target
        result["asr"] = float(hits.float().mean())
        result["asr_non_target"] = (
            float(hits[non_target].float().mean()) if non_target.any() else None
        )

    output_dir = Path(config["output_dir"]) / result["experiment_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "scores.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "label", "predicted", "is_poisoned", "score"])
        for path, label, predicted, score in zip(paths, labels, clean_pred, clean_scores):
            writer.writerow([path, label.item(), predicted.item(), 0, float(score)])
        for path, label, predicted, score in zip(
            poisoned_paths, poisoned_labels, poisoned_pred, poisoned_scores
        ):
            writer.writerow([path, label.item(), predicted.item(), 1, float(score)])
    return result
