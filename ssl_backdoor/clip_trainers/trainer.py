"""CLIP image-text contrastive trainer: seed/DDP initialization, AMP, gradient accumulation, checkpoint management."""
import logging
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ssl_backdoor.datasets.image_caption import ImageCaptionDataset
from .losses import ClipLoss
from .modeling import build_clip

LOGIT_SCALE_MAX = math.log(100.0)
NORM_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CLIPTrainer:
    def __init__(self, config, rank=0, world_size=1, device_id=0):
        self.cfg = config
        self.train_cfg = config.get('train', {})
        self.dist_cfg = config.get('distributed', {})
        self.eval_cfg = config.get('eval', {})
        self.rank, self.world_size = rank, world_size
        self.distributed = world_size > 1
        set_seed(config.get('seed', 42) + rank)

        torch.cuda.set_device(device_id)
        self.device = torch.device('cuda', device_id)
        if self.distributed and not dist.is_initialized():
            dist.init_process_group(
                backend=self.dist_cfg.get('backend', 'nccl'),
                init_method=self.dist_cfg.get('init_method', 'env://'),
                rank=rank,
                world_size=world_size)

        self.exp_dir = os.path.join(config['save_folder_root'], config['experiment_id'])
        os.makedirs(self.exp_dir, exist_ok=True)
        self.logger = self._build_logger()

        self.model, self.processor = build_clip(config['model'])
        self.model.to(self.device)
        if self.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[device_id])

        data_cfg = config['data']
        train_file = data_cfg.get('train_csv') or data_cfg.get('train_file')
        if not train_file:
            raise ValueError('data.train_csv must be configured')
        self.train_loader, self.train_sampler = self._build_loader(train_file, train=True)
        self.val_loader = None
        val_file = data_cfg.get('val_csv') or data_cfg.get('val_file')
        if val_file and self.eval_cfg.get('enabled', True):
            self.val_loader, _ = self._build_loader(val_file, train=False)

        self.criterion = ClipLoss()
        self.accum_steps = self.train_cfg.get('grad_accum_steps',
                                               config.get('accum_steps', 1))
        self.amp = self.train_cfg.get('amp', config.get('amp', True))
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(math.ceil(len(self.train_loader) / self.accum_steps))
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)

        self.start_epoch = 0
        if config.get('resume'):
            self._load_checkpoint(config['resume'])

    # ---------- Build ----------
    def _build_logger(self):
        logger = logging.getLogger(f'clip_trainer_rank{self.rank}')
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if self.rank == 0 and not logger.handlers:
            fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            for handler in (logging.StreamHandler(),
                            logging.FileHandler(os.path.join(self.exp_dir, 'train.log'))):
                handler.setFormatter(fmt)
                logger.addHandler(handler)
        return logger

    def _build_loader(self, csv_path, train):
        data_cfg = self.cfg['data']
        dataset = ImageCaptionDataset(
            csv_path, self.processor,
            image_key=data_cfg.get('image_key', 'image'),
            caption_key=data_cfg.get('caption_key', 'caption'),
            delimiter=data_cfg.get('delimiter', ','),
            image_root=data_cfg.get('image_root'),
            max_length=data_cfg.get('max_length'))
        sampler = DistributedSampler(dataset, shuffle=train) if self.distributed else None
        loader = DataLoader(
            dataset, batch_size=self.train_cfg.get('batch_size', self.cfg.get('batch_size', 64)),
            shuffle=(train and sampler is None), sampler=sampler,
            num_workers=self.train_cfg.get('workers', self.cfg.get('workers', 4)),
            pin_memory=True, drop_last=train)
        return loader, sampler

    def _build_optimizer(self):
        model = self.model.module if self.distributed else self.model
        norm_param_ids = {id(p) for m in model.modules() if isinstance(m, NORM_TYPES)
                          for p in m.parameters(recurse=False)}
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith('.bias') or id(param) in norm_param_ids or 'logit_scale' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        opt_cfg = self.cfg.get('optimizer', self.train_cfg)
        return torch.optim.AdamW(
            [{'params': decay, 'weight_decay': opt_cfg.get('weight_decay', 0.1)},
             {'params': no_decay, 'weight_decay': 0.0}],
            lr=opt_cfg.get('lr', self.train_cfg.get('lr', 1e-4)),
            betas=tuple(opt_cfg.get('betas', (0.9, 0.98))),
            eps=opt_cfg.get('eps', 1e-6))

    def _build_scheduler(self, steps_per_epoch):
        total_steps = self.train_cfg.get('epochs', self.cfg.get('epochs', 1)) * steps_per_epoch
        warmup_steps = self.train_cfg.get('warmup_steps', self.cfg.get('warmup_steps',
                                    int(self.cfg.get('warmup_epochs', 0) * steps_per_epoch))
                                    )

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ---------- checkpoint ----------
    def _save_checkpoint(self, epoch):
        model = self.model.module if self.distributed else self.model
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'config': self.cfg,
        }
        torch.save(state, os.path.join(self.exp_dir, 'checkpoint.pth'))
        save_freq = self.train_cfg.get('save_interval', self.cfg.get('save_freq', 1))
        if (epoch + 1) % save_freq == 0:
            torch.save(state, os.path.join(self.exp_dir, f'checkpoint_epoch{epoch + 1}.pth'))

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        model = self.model.module if self.distributed else self.model
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler'])
        self.start_epoch = ckpt.get('epoch', 0)
        if self.rank == 0:
            self.logger.info(f'Resumed from {path}, start_epoch={self.start_epoch}')

    # ---------- Train/Validate ----------
    def _forward_loss(self, batch):
        out = self.model(
            pixel_values=batch['pixel_values'].to(self.device, non_blocking=True),
            input_ids=batch['input_ids'].to(self.device, non_blocking=True),
            attention_mask=batch['attention_mask'].to(self.device, non_blocking=True))
        return self.criterion(out['image_embeds'], out['text_embeds'], out['logit_scale'])

    def train_one_epoch(self, epoch):
        self.model.train()
        logit_scale = (self.model.module if self.distributed else self.model).logit_scale
        log_interval = self.train_cfg.get('log_interval', self.cfg.get('log_interval', 50))
        total_loss, num_batches = 0.0, len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(self.train_loader):
            with torch.autocast('cuda', enabled=self.amp):
                loss = self._forward_loss(batch)
            self.scaler.scale(loss / self.accum_steps).backward()

            if (i + 1) % self.accum_steps == 0 or (i + 1) == num_batches:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                with torch.no_grad():
                    logit_scale.clamp_(max=LOGIT_SCALE_MAX)

            total_loss += loss.item()
            if self.rank == 0 and (i % log_interval == 0 or i + 1 == num_batches):
                self.logger.info(
                    f'epoch {epoch} [{i + 1}/{num_batches}] loss={loss.item():.4f} '
                    f'lr={self.optimizer.param_groups[0]["lr"]:.2e} '
                    f'logit_scale={logit_scale.exp().item():.2f}')
        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            raise ValueError('data.val_file not configured, cannot validate')
        self.model.eval()
        total_loss, num_batches = 0.0, len(self.val_loader)
        for batch in self.val_loader:
            with torch.autocast('cuda', enabled=self.amp):
                total_loss += self._forward_loss(batch).item()
        avg = torch.tensor(total_loss / max(1, num_batches), device=self.device)
        if self.distributed:
            dist.all_reduce(avg, op=dist.ReduceOp.AVG)
        if self.rank == 0:
            self.logger.info(f'validation loss={avg.item():.4f}')
        return avg.item()

    def train(self):
        epochs = self.train_cfg.get('epochs', self.cfg.get('epochs', 1))
        for epoch in range(self.start_epoch, epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            train_loss = self.train_one_epoch(epoch)
            if self.rank == 0:
                self.logger.info(f'epoch {epoch} average training loss={train_loss:.4f}')
            val_interval = self.eval_cfg.get('validation_interval', 1)
            if self.val_loader is not None and (epoch + 1) % val_interval == 0:
                self.validate()
            if self.rank == 0:
                self._save_checkpoint(epoch)
        self.finish()

    def finish(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
