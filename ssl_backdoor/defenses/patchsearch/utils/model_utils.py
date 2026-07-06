"""PatchSearch utility implementation."""

import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as models
import sys

from ssl_backdoor.utils.model_utils import load_checkpoint


def load_weights(model, wts_path):
    """PatchSearch utility implementation."""
    state_dict = load_checkpoint(wts_path)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    state_dict = {k.replace('encoder_q.', ''): v for k, v in state_dict.items()}
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
    state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    
    model.load_state_dict(state_dict, strict=False)
    return model


def get_model(arch, wts_path, dataset_name):
    """PatchSearch utility implementation."""
    if 'moco' in arch:
        model = models.__dict__[arch.replace('moco_', '')]()
        if 'imagenet' not in dataset_name:
            print("Using custom conv1 for small datasets")
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if dataset_name == "cifar10" or dataset_name == "cifar100":
            print("Using custom maxpool for cifar datasets")
            model.maxpool = nn.Identity()
        model.fc = nn.Sequential()

        sd = torch.load(wts_path)['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}

        sd = {k: v for k, v in sd.items() if 'encoder_q' in k or 'base_encoder' in k or 'backbone' in k or 'encoder' in k}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}

        sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
        sd = {k.replace('base_encoder.', ''): v for k, v in sd.items()}
        sd = {k.replace('backbone.', ''): v for k, v in sd.items()}
        sd = {k.replace('encoder.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)

    elif 'resnet' in arch:
        model = models.__dict__[arch]()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
        
    else:
        raise ValueError('arch not found: ' + arch)

    model = model.eval()

    return model


def get_feats(model, loader):
    """PatchSearch utility implementation."""
    model = nn.DataParallel(model).cuda()
    model.eval()
    feats, labels, indices, is_poisoned = [], [], [], []
    for data in tqdm(loader):
        if len(data) == 4:
            images, targets, is_p, inds = data
        else:
            images, targets = data
        with torch.no_grad():
            feats.append(model(images.cuda()).cpu())
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