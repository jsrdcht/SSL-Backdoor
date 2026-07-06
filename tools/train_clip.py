"""CLIP image-text contrastive training entry point: read a YAML config and support single-GPU and DDP."""
import argparse
import os

import torch.multiprocessing as mp
import yaml

from ssl_backdoor.clip_trainers.trainer import CLIPTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='CLIP image-text contrastive training')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    parser.add_argument('--experiment_id', type=str, default=None, help='Override experiment_id in the config')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--device_ids', type=str, default=None, help='GPU id list, such as 0,1,2,3')
    parser.add_argument('--eval-only', '--eval_only', dest='eval_only', action='store_true',
                        help='Compute loss on the validation set only')
    return parser.parse_args()


def worker(local_rank, world_size, device_ids, config):
    trainer = CLIPTrainer(config, rank=local_rank, world_size=world_size,
                          device_id=device_ids[local_rank])
    if config.get('eval_only'):
        trainer.validate()
        trainer.finish()
    else:
        trainer.train()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.experiment_id:
        config['experiment_id'] = args.experiment_id
    if args.resume:
        config['resume'] = args.resume
    config['eval_only'] = args.eval_only
    if args.device_ids:
        device_ids = [int(x) for x in args.device_ids.split(',')]
    else:
        device_ids = config.get('distributed', {}).get('device_ids', [0])
    config.setdefault('distributed', {})['device_ids'] = device_ids

    exp_dir = os.path.join(config['save_folder_root'], config['experiment_id'])
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'final_config.yaml'), 'w') as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    world_size = len(device_ids)
    if world_size > 1:
        mp.spawn(worker, args=(world_size, device_ids, config), nprocs=world_size)
    else:
        worker(0, 1, device_ids, config)


if __name__ == '__main__':
    main()
