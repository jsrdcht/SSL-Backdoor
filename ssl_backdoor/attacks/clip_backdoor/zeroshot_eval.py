"""Zero-shot classification + backdoor ASR evaluation.

Reuses build_clip to construct CLIPModel consistent with training, loads training-saved state_dict; text prototypes
assembled from classes.py classes+templates; ASR applies trigger consistent with training to same batch images then determines target class hits.
"""
import csv

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ssl_backdoor.clip_trainers.modeling import build_clip
from ssl_backdoor.datasets.attacker.triggers import apply_static_trigger
from ssl_backdoor.datasets.pre_resize import pre_resize_image, resolve_pre_resize
from .caption_targets import load_classes_config, load_templates, resolve_target_index
from .utils import build_trigger_args


class _ZeroShotDataset(Dataset):
    """Read labels.csv (image,label), optionally apply trigger to each image."""

    def __init__(
        self,
        labels_csv,
        processor,
        trigger_args=None,
        trigger_path=None,
        pre_resize=False,
        pre_resize_size=None,
    ):
        self.processor = processor
        self.trigger_args = trigger_args
        self.trigger_path = trigger_path
        self.pre_resize = pre_resize
        self.pre_resize_size = pre_resize_size
        with open(labels_csv, newline="") as f:
            self.samples = [(r["image"], int(r["label"])) for r in csv.DictReader(f)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = pre_resize_image(image, self.pre_resize, self.pre_resize_size)
        if self.trigger_args is not None:
            image = apply_static_trigger(image, self.trigger_args, trigger=self.trigger_path)
        pixel = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixel, label


def _load_model(cfg, device):
    model, processor = build_clip(cfg["model"])
    ckpt = cfg.get("checkpoint")
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state))
        print(f"[eval] Loaded checkpoint: {ckpt}")
    return model.to(device).eval(), processor


@torch.no_grad()
def _text_prototypes(model, processor, classes, templates, device):
    """For each class: normalize template text features then take mean, normalize again as text prototype. Returns [dim, num_classes]."""
    embeddings = []
    for c in classes:
        tokens = processor(text=[t(c) for t in templates], padding="max_length",
                           truncation=True, max_length=77, return_tensors="pt").to(device)
        feats = model.model.get_text_features(input_ids=tokens["input_ids"],
                                              attention_mask=tokens["attention_mask"])
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.mean(dim=0)
        embeddings.append(feats / feats.norm())
    return torch.stack(embeddings, dim=1).to(device)


@torch.no_grad()
def _predict_topk(model, loader, text_proto, device, topk):
    """Return [N, max_topk] predicted class indices and true labels [N]."""
    all_ranks, all_labels = [], []
    for pixel, label in loader:
        feats = model.model.get_image_features(pixel_values=pixel.to(device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = feats @ text_proto
        all_ranks.append(logits.topk(max(topk), dim=1)[1].cpu())
        all_labels.append(label)
    return torch.cat(all_ranks), torch.cat(all_labels)


def evaluate(cfg):
    """Execute clean zero-shot + ASR evaluation, return results dict."""
    pre_resize, pre_resize_size = resolve_pre_resize(cfg)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    topk = cfg.get("topk", [1, 5])
    model, processor = _load_model(cfg, device)

    config = load_classes_config(cfg["classes_path"])
    classes = config["classes"]
    # Default aligns with CleanCLIP: if eval_templates not explicitly specified, directly use full template pool from classes.py.
    templates_path = cfg.get("eval_templates") or cfg["classes_path"]
    templates = load_templates(
        templates_path,
        cfg.get("num_eval_templates"),
        cfg.get("seed", 0),
        fallback_to_default=False,
    )
    text_proto = _text_prototypes(model, processor, classes, templates, device)

    batch_size, workers = cfg.get("batch_size", 128), cfg.get("workers", 8)
    labels_csv = cfg["labels_csv"]
    target_index = resolve_target_index(cfg["attack_target"], classes)

    # clean zero-shot
    clean_ds = _ZeroShotDataset(
        labels_csv,
        processor,
        pre_resize=pre_resize,
        pre_resize_size=pre_resize_size,
    )
    clean_loader = DataLoader(clean_ds, batch_size=batch_size, num_workers=workers, pin_memory=True)
    ranks, labels = _predict_topk(model, clean_loader, text_proto, device, topk)
    results = {f"zeroshot_top{k}": (ranks[:, :k] == labels[:, None]).any(1).float().mean().item()
               for k in topk}

    # ASR: apply trigger to same batch images, determine target class hits; also report hits after excluding samples already belonging to target class
    trigger_args = build_trigger_args(cfg["trigger"])
    bd_ds = _ZeroShotDataset(
        labels_csv,
        processor,
        trigger_args,
        cfg["trigger"].get("trigger_path"),
        pre_resize=pre_resize,
        pre_resize_size=pre_resize_size,
    )
    bd_loader = DataLoader(bd_ds, batch_size=batch_size, num_workers=workers, pin_memory=True)
    bd_ranks, bd_labels = _predict_topk(model, bd_loader, text_proto, device, topk)
    non_target = bd_labels != target_index
    for k in topk:
        hit = (bd_ranks[:, :k] == target_index).any(1)
        results[f"asr_top{k}"] = hit.float().mean().item()
        results[f"asr_top{k}_nontarget"] = hit[non_target].float().mean().item()

    results["target_index"] = target_index
    results["num_samples"] = int(labels.numel())
    return results
