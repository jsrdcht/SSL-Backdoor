"""Optimize a BadCLIP patch for later use with trigger_insert=patch."""
import argparse

import yaml

from ssl_backdoor.attacks.badclip.trigger_optimizer import BadCLIPTriggerOptimizer


def parse_args():
    p = argparse.ArgumentParser(description="BadCLIP trigger optimization")
    p.add_argument("--config", required=True, help="YAML containing model and trigger_optimization sections")
    p.add_argument("--device", default="cuda")
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    opt_cfg = cfg["trigger_optimization"]
    if not opt_cfg.get("enabled", True):
        print("[badclip] trigger_optimization.enabled=False, skipping optimization")
        return

    optimizer = BadCLIPTriggerOptimizer(cfg["model"], device=args.device)
    optimizer.optimize(opt_cfg)

if __name__ == "__main__":
    main()
