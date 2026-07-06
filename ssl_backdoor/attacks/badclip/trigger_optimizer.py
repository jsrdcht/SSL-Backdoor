"""BadCLIP trigger optimizer: optimizes trigger patches via dual constraints (CVPR 2024).

Core Algorithm (reproduces BadCLIP/src/embeding_optimize_patch.py's optimize_patch):
1. Paste learnable patch in [0,1] pixel space, then apply CLIP normalization before feeding to frozen CLIP.
2. Dual losses with gradient descent on patch pixels:
   - Image-text alignment (InfoNCE): makes triggered image embedding close to target class text embedding;
   - Triplet loss: anchor=triggered image, positive=target class image, negative=clean images in batch,
     makes triggered image close to target class in visual embedding space and far from original semantics.
3. Patch is clamped back to [0,1] after each step, finally saved as PNG for standard patch/blend trigger deployment.

Notes:
- ImageCaptionDataset returns normalized pixel_values, cannot directly paste [0,1] patch;
  so we use _RawImageDataset to return [0,1] raw pixels, normalization done inside optimizer.
- Target texts sampled from caption_targets template pool (provides batch diversity),
  ignoring the original project's natural language EDA augmentation.
"""
import csv
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ssl_backdoor.attacks.clip_backdoor.caption_targets import build_target_caption, load_templates
from ssl_backdoor.clip_trainers.modeling import build_clip


class _RawImageDataset(Dataset):
    """Reads CSV image column, returns [0,1] raw pixel tensors (resize + center crop to model input resolution).

    No CLIP normalization, no caption loading: patch needs to be pasted in [0,1] space, normalization delegated to optimizer.
    """

    def __init__(self, csv_path, image_key='image', delimiter=',', image_root=None, resolution=224):
        self.image_root = image_root or os.path.dirname(os.path.abspath(csv_path))
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if image_key not in (reader.fieldnames or []):
                raise ValueError(f"CSV {csv_path} missing column {image_key!r}, available columns: {reader.fieldnames}")
            self.paths = [row[image_key] for row in reader]
        if not self.paths:
            raise ValueError(f"CSV {csv_path} contains no samples")
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image_path = path if os.path.isabs(path) else os.path.join(self.image_root, path)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)


class BadCLIPTriggerOptimizer:
    """BadCLIP trigger optimizer: optimizes trigger patches with dual losses."""

    def __init__(self, model_cfg, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model, self.processor = build_clip(model_cfg)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.clip = self.model.model  # Underlying HuggingFace CLIPModel

        ip = self.processor.image_processor
        crop = ip.crop_size
        self.resolution = crop['height'] if isinstance(crop, dict) else crop
        self._mean = torch.tensor(ip.image_mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(ip.image_std, device=self.device).view(1, 3, 1, 1)

    # ---- Trigger initialization and pasting ----
    def _init_patch(self, patch_size, init_mode, seed):
        if init_mode == 'const':
            patch = torch.full((1, 3, patch_size, patch_size), 0.5, dtype=torch.float32)
        else:
            rng = np.random.RandomState(seed)
            arr = np.clip(rng.normal(0.5, 0.25, (1, 3, patch_size, patch_size)), 0, 1)
            patch = torch.from_numpy(arr.astype(np.float32))
        return patch.to(self.device).requires_grad_(True)

    def _embed_patch(self, images, patch, location, patch_size, blend_ratio):
        """Paste patch on [0,1] images. location in {'middle','center'} centers the patch, 'blended' blends over entire image."""
        p = torch.clamp(patch, 0.0, 1.0)
        if location == 'blended':
            p = F.interpolate(p, size=images.shape[-2:], mode='bilinear', align_corners=False)
            return torch.clamp(blend_ratio * p + (1 - blend_ratio) * images, 0.0, 1.0)
        out = images.clone()
        h, w = images.shape[-2:]
        s0, s1 = (h - patch_size) // 2, (w - patch_size) // 2
        out[:, :, s0:s0 + patch_size, s1:s1 + patch_size] = p
        return out

    # ---- Feature extraction ----
    def _normalize(self, images):
        return (images - self._mean) / self._std

    def _image_embeds(self, norm_images):
        feats = self.clip.get_image_features(pixel_values=norm_images)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _text_embeds(self, captions):
        tok = self.processor(text=captions, padding='max_length', truncation=True,
                             max_length=77, return_tensors='pt').to(self.device)
        feats = self.clip.get_text_features(input_ids=tok['input_ids'],
                                            attention_mask=tok['attention_mask'])
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _extract_image_embeds(self, csv_path, opt_cfg, batch_size):
        ds = _RawImageDataset(csv_path, image_key=opt_cfg.get('image_key', 'image'),
                              delimiter=opt_cfg.get('delimiter', ','),
                              image_root=opt_cfg.get('image_root'), resolution=self.resolution)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=opt_cfg.get('workers', 4), pin_memory=True)
        embeds = []
        for imgs in loader:
            imgs = imgs.to(self.device, non_blocking=True)
            embeds.append(self._image_embeds(self._normalize(imgs)))
        return torch.cat(embeds, dim=0)

    # ---- Loss computation ----
    def _compute_losses(self, trigger_embeds, text_embeds, clean_embeds, pos_embeds, opt_cfg):
        losses = {}
        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * trigger_embeds @ text_embeds.t()
        target = torch.arange(logits.size(0), device=self.device)
        losses['img_text'] = (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2

        if pos_embeds is not None:
            margin = opt_cfg.get('triplet_margin', 1.0)
            n = min(trigger_embeds.size(0), pos_embeds.size(0), clean_embeds.size(0))
            anchor = trigger_embeds[:n]
            pos = pos_embeds[torch.randperm(pos_embeds.size(0), device=self.device)[:n]]
            neg = clean_embeds[torch.randperm(clean_embeds.size(0), device=self.device)[:n]]
            if opt_cfg.get('use_cosine_triplet', False):
                tri = F.relu(F.cosine_similarity(anchor, neg) -
                             F.cosine_similarity(anchor, pos) + margin).mean()
            else:
                tri = F.triplet_margin_loss(anchor, pos, neg, margin=margin)
            losses['triplet'] = opt_cfg.get('lambda_triplet', 500) * tri
        return losses

    # ---- Main pipeline ----
    def optimize(self, opt_cfg):
        """Execute trigger optimization, returns optimized patch tensor [1,3,ps,ps]."""
        seed = opt_cfg.get('seed', 42)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        patch_size = opt_cfg.get('patch_size', 16)
        location = opt_cfg.get('patch_location', 'middle')
        if location not in ('middle', 'center', 'blended'):
            raise ValueError(f"patch_location must be middle/center/blended, got: {location!r}")
        blend_ratio = opt_cfg.get('blend_ratio', 0.2)
        batch_size = opt_cfg.get('batch_size', 64)
        image_root = opt_cfg.get('image_root')

        patch = self._init_patch(patch_size, opt_cfg.get('patch_init', 'random'), seed)
        optimizer = torch.optim.Adam([patch], lr=opt_cfg.get('learning_rate', 0.001))

        train_ds = _RawImageDataset(opt_cfg['train_data_csv'], image_key=opt_cfg.get('image_key', 'image'),
                                    delimiter=opt_cfg.get('delimiter', ','), image_root=image_root,
                                    resolution=self.resolution)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=opt_cfg.get('workers', 4), pin_memory=True, drop_last=True)

        # Target text template pool: provides batch diversity, ignoring natural language EDA augmentation.
        templates = load_templates(opt_cfg.get('caption_templates'), opt_cfg.get('num_templates'), seed)
        target_label = opt_cfg['target_label']
        cap_rng = random.Random(seed)

        # Positive samples (target class images) embeddings: pre-extract once; degrades to pure image-text alignment if no positive CSV.
        pos_embeds = None
        if opt_cfg.get('positive_samples_csv'):
            pos_embeds = self._extract_image_embeds(opt_cfg['positive_samples_csv'], opt_cfg, batch_size)

        num_steps = opt_cfg.get('num_steps', 50)
        prog_interval = opt_cfg.get('prog_interval', 5)
        print(f"[badclip] Optimizing trigger: {num_steps} epochs x {len(train_loader)} batches, "
              f"target={target_label}, patch={patch_size}/{location}, "
              f"lambda_triplet={opt_cfg.get('lambda_triplet', 500)}, triplet={'on' if pos_embeds is not None else 'off'}")

        for epoch in range(num_steps):
            epoch_losses = {}
            for images in train_loader:
                images = images.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                trigger_img = self._embed_patch(images, patch, location, patch_size, blend_ratio)
                trigger_embeds = self._image_embeds(self._normalize(trigger_img))
                with torch.no_grad():
                    clean_embeds = self._image_embeds(self._normalize(images))
                captions = [build_target_caption(target_label, templates, cap_rng)
                            for _ in range(images.size(0))]
                text_embeds = self._text_embeds(captions)

                losses = self._compute_losses(trigger_embeds, text_embeds, clean_embeds, pos_embeds, opt_cfg)
                total = sum(losses.values())
                total.backward()
                optimizer.step()
                with torch.no_grad():
                    patch.clamp_(0.0, 1.0)

                for k, v in losses.items():
                    epoch_losses.setdefault(k, []).append(v.item())

            if (epoch + 1) % prog_interval == 0 or epoch == num_steps - 1:
                summary = ", ".join(f"{k}={np.mean(v):.4f}" for k, v in epoch_losses.items())
                print(f"[badclip] Epoch {epoch + 1}/{num_steps}: {summary}")

        output_path = opt_cfg.get('output_trigger_path')
        if output_path:
            self.save_trigger(patch, output_path)
            print(f"[badclip] Trigger saved to: {output_path}")
        return patch

    def save_trigger(self, patch, output_path):
        """Save patch as PNG (for standard patch/blend trigger deployment)."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
        arr = torch.clamp(patch.squeeze(0).detach().cpu(), 0, 1).mul(255).round()
        arr = arr.numpy().astype(np.uint8).transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
        cv2.imwrite(output_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
