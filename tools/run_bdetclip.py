"""Command-line entry point for BDetCLIP."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from ssl_backdoor.defenses.bdetclip import run_bdetclip
from ssl_backdoor.ssl_trainers.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="BDetCLIP test-time backdoor detection")
    parser.add_argument("--config", required=True, help="BDetCLIP YAML config")
    parser.add_argument("--device", help="override the configured device")
    parser.add_argument("--evaluation-samples", type=int, help="override evaluation set size")
    parser.add_argument("--reference-samples", type=int, help="override reference set size")
    parser.add_argument("--output-dir", help="override the result root directory")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if not isinstance(config, dict):
        raise TypeError("the BDetCLIP config must be a mapping")
    if args.device:
        config["device"] = args.device
    if args.evaluation_samples is not None:
        config.setdefault("data", {})["evaluation_samples"] = args.evaluation_samples
    if args.reference_samples is not None:
        config.setdefault("data", {})["reference_samples"] = args.reference_samples
    if args.output_dir:
        config["output_dir"] = args.output_dir

    experiment_id = config.get("experiment_id", "bdetclip")
    output_dir = Path(config.get("output_dir", "results/defense/bdetclip")) / experiment_id
    config["experiment_id"], config["output_dir"] = experiment_id, str(output_dir.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "run.log")],
        force=True,
    )
    (output_dir / "final_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    logging.info("Loaded config: %s", args.config)
    logging.info("Output directory: %s", output_dir)
    result = run_bdetclip(config)
    logging.info("Detection complete:\n%s", json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
