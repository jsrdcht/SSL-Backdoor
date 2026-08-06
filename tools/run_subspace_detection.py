"""Command-line entry point for Subspace Detection."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from ssl_backdoor.defenses.subspace_detection import run_subspace_detection
from ssl_backdoor.ssl_trainers.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Test-time CLIP Subspace Detection")
    parser.add_argument("--config", required=True, help="defense YAML config")
    parser.add_argument("--device", help="override the configured device")
    parser.add_argument("--num-samples", type=int, help="override evaluation set size")
    parser.add_argument("--reference-samples", type=int, help="override reference set size")
    parser.add_argument("--output-dir", help="override the result root directory")
    return parser.parse_args()


def _setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "run.log")],
        force=True,
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    if not isinstance(config, dict):
        raise TypeError("the Subspace Detection config must be a mapping")
    if args.device:
        config["device"] = args.device
    if args.num_samples is not None:
        config.setdefault("data", {})["num_samples"] = args.num_samples
    if args.reference_samples is not None:
        config.setdefault("data", {})["reference_samples"] = args.reference_samples
    if args.output_dir:
        config["output_dir"] = args.output_dir

    experiment_id = config.get("experiment_id", "subspace_detection")
    result_root = config.get("output_dir", "results/defense/subspace_detection")
    output_dir = Path(result_root) / experiment_id
    config["experiment_id"], config["output_dir"] = experiment_id, str(output_dir.parent)
    _setup_logging(output_dir)
    (output_dir / "final_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    logging.info("Loaded config: %s", args.config)
    logging.info("Output directory: %s", output_dir)
    result = run_subspace_detection(config)
    logging.info("Detection complete:\n%s", json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
