"""End-to-end CLIP backdoor entry point: generate poisoned data -> train -> evaluate.

Each stage can be triggered separately with --stage (poison / train / eval / all).
Training uses the same CLIPTrainer from the clean-base tools/train_clip.py; this
script only injects the poisoned CSV into the training config.
"""
import argparse
import json
import os

import yaml

from ssl_backdoor.attacks.clip_backdoor.poison_generator import generate_poison
from ssl_backdoor.attacks.clip_backdoor.zeroshot_eval import evaluate as run_eval


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="CLIP backdoor generation/training/evaluation")
    p.add_argument("--poison_config", type=str, help="YAML config for poisoned data generation")
    p.add_argument("--train_config", type=str, help="Training YAML config (reuses the train_clip structure)")
    p.add_argument("--eval_config", type=str, help="Evaluation YAML config")
    p.add_argument("--stage", choices=["poison", "train", "eval", "all"], default="all")
    p.add_argument("--device_ids", type=str, default=None)
    return p.parse_args()


def _train_worker(rank, world_size, device_ids, config):
    from ssl_backdoor.clip_trainers.trainer import CLIPTrainer

    trainer = CLIPTrainer(config, rank=rank, world_size=world_size, device_id=device_ids[rank])
    trainer.train()


def do_train(train_config, poisoned_csv, device_ids):
    # Delayed import: training depends on CUDA/DDP, while evaluation or generation does not need it.
    import torch.multiprocessing as mp

    config = _load(train_config)
    if poisoned_csv:
        config["data"]["train_csv"] = poisoned_csv
    device_ids = [int(x) for x in device_ids.split(",")] if device_ids \
        else config.get("distributed", {}).get("device_ids", [0])
    config.setdefault("distributed", {})["device_ids"] = device_ids

    exp_dir = os.path.join(config["save_folder_root"], config["experiment_id"])
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "final_config.yaml"), "w") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    world_size = len(device_ids)
    if world_size > 1:
        mp.spawn(_train_worker, args=(world_size, device_ids, config), nprocs=world_size)
    else:
        _train_worker(0, 1, device_ids, config)
    return os.path.join(exp_dir, "checkpoint.pth")


def main():
    args = parse_args()
    poisoned_csv = None

    if args.stage in ("poison", "all") and args.poison_config:
        poisoned_csv = generate_poison(_load(args.poison_config))

    ckpt = None
    if args.stage in ("train", "all") and args.train_config:
        ckpt = do_train(args.train_config, poisoned_csv, args.device_ids)
        print(f"[train] checkpoint: {ckpt}")

    if args.stage in ("eval", "all") and args.eval_config:
        eval_cfg = _load(args.eval_config)
        if ckpt and not eval_cfg.get("checkpoint"):
            eval_cfg["checkpoint"] = ckpt
        results = run_eval(eval_cfg)
        print("[eval] results:\n" + json.dumps(results, indent=2, ensure_ascii=False))
        out = eval_cfg.get("results_json")
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
