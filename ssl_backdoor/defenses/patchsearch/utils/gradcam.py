"""PatchSearch utility implementation."""

import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(tensor, height=14, width=14):
    """PatchSearch utility implementation."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def run_gradcam(arch, model, inp, targets=None):
    """PatchSearch utility implementation."""
    if 'vit' in arch:
        return run_vit_gradcam(model, [model.blocks[-1].norm1], inp, targets)
    else:
        return run_cnn_gradcam(model, [model.layer4], inp, targets)


def run_cnn_gradcam(model, target_layers, inp, targets=None):
    """PatchSearch utility implementation."""
    params_to_restore = []
    for layer in target_layers:
        for param in layer.parameters():
            if not param.requires_grad:
                params_to_restore.append((param, param.requires_grad))
                param.requires_grad_(True)
    
    try:
        with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
            cam.batch_size = 32
            grayscale_cam, out = cam(input_tensor=inp, targets=targets)
            return grayscale_cam, out
    finally:
        for param, orig_requires_grad in params_to_restore:
            param.requires_grad_(orig_requires_grad)


def run_vit_gradcam(model, target_layers, inp, targets=None):
    """PatchSearch utility implementation."""
    with GradCAM(model=model, target_layers=target_layers,
            reshape_transform=reshape_transform, use_cuda=True) as cam:
        cam.batch_size = 32
        grayscale_cam, out = cam(input_tensor=inp, targets=targets)
        return grayscale_cam, out 