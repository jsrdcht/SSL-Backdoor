"""PatchSearch utility implementation."""

import torch
import torch.nn as nn
from tqdm import tqdm
from ssl_backdoor.utils.model_utils import get_backbone_model


def get_model(arch, wts_path, dataset_name):
    """Load a frozen encoder, accepting legacy ``moco_resnet*`` names."""
    return get_backbone_model(
        arch.removeprefix('moco_'), wts_path, device='cpu',
        dataset=dataset_name, freeze_backbone=True,
    ).eval()


def get_feats(model, loader):
    """PatchSearch utility implementation."""
    device = next(model.parameters()).device
    if device.type == "cuda":
        model = nn.DataParallel(model)
    model.eval()
    feats, labels, indices, is_poisoned = [], [], [], []
    for data in tqdm(loader):
        if len(data) == 4:
            images, targets, is_p, inds = data
        else:
            images, targets = data
        with torch.no_grad():
            feats.append(model(images.to(device)).cpu())
            labels.append(targets)
            indices.append(inds)
            is_poisoned.append(is_p)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    indices = torch.cat(indices)
    is_poisoned = torch.cat(is_poisoned)
    feats /= feats.norm(2, dim=-1, keepdim=True)
    return feats, labels, is_poisoned, indices


def get_channels(arch):
    """PatchSearch utility implementation."""
    if 'resnet50' in arch:
        c = 2048
    elif 'resnet18' in arch:
        c = 512
    else:
        raise ValueError('arch not found: ' + arch)
    return c
