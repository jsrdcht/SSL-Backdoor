"""PatchSearch utility implementation."""

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from PIL import Image
from .gradcam import run_gradcam
from .dataset import denormalize


def paste_patch(inputs, patch):
    """PatchSearch utility implementation."""
    B = inputs.shape[0]
    inp_w = inputs.shape[-1]
    window_w = patch.shape[-1]
    ij = torch.randint(low=0, high=(inp_w - window_w), size=(B, 2))
    i, j = ij[:, 0], ij[:, 1]
    s = torch.arange(window_w, device=inputs.device)
    ri = i.view(B, 1).repeat(1, window_w)
    rj = j.view(B, 1).repeat(1, window_w)
    sri, srj = ri + s, rj + s
    xi = sri.view(B, window_w, 1).repeat(1, 1, window_w)
    xj = srj.view(B, 1, window_w).repeat(1, window_w, 1)
    inds = xi * inp_w + xj
    inds = inds.unsqueeze(1).repeat((1, 3, 1, 1)).view(B, 3, -1)
    patch = patch.reshape(3, -1).unsqueeze(0).repeat(B, 1, 1)
    inputs = inputs.reshape(B, 3, -1)
    inputs.scatter_(dim=2, index=inds, src=patch)
    inputs = inputs.reshape(B, 3, inp_w, inp_w)
    return inputs


def block_max_window(cam_images, inputs, window_w=30):
    """PatchSearch utility implementation."""
    B, _, inp_w = cam_images.shape
    grayscale_cam = torch.from_numpy(cam_images)
    inputs = inputs.clone()
    sum_conv = torch.ones((1, 1, window_w, window_w))
    sums_cam = F.conv2d(grayscale_cam.unsqueeze(1), sum_conv)
    flat_sums_cam = sums_cam.view(B, -1)
    ij = flat_sums_cam.argmax(dim=-1)
    sums_cam_w = sums_cam.shape[-1]
    i, j = ij // sums_cam_w, ij % sums_cam_w
    s = torch.arange(window_w, device=inputs.device)
    ri = i.view(B, 1).repeat(1, window_w)
    rj = j.view(B, 1).repeat(1, window_w)
    sri, srj = ri + s, rj + s
    xi = sri.view(B, window_w, 1).repeat(1, 1, window_w)
    xj = srj.view(B, 1, window_w).repeat(1, window_w, 1)
    inds = xi * inp_w + xj
    inds = inds.unsqueeze(1).repeat((1, 3, 1, 1)).view(B, 3, -1)
    inputs = inputs.reshape(B, 3, -1)
    inputs.scatter_(dim=2, index=inds, value=0)
    inputs = inputs.reshape(B, 3, inp_w, inp_w)
    return inputs


def extract_max_window(cam_images, inputs, window_w=30):
    """PatchSearch utility implementation."""
    B, _, inp_w = cam_images.shape
    grayscale_cam = torch.from_numpy(cam_images)
    inputs = inputs.clone()
    sum_conv = torch.ones((1, 1, window_w, window_w))
    sums_cam = F.conv2d(grayscale_cam.unsqueeze(1), sum_conv)
    flat_sums_cam = sums_cam.view(B, -1)
    ij = flat_sums_cam.argmax(dim=-1)
    sums_cam_w = sums_cam.shape[-1]
    i, j = ij // sums_cam_w, ij % sums_cam_w
    s = torch.arange(window_w, device=inputs.device)
    ri = i.view(B, 1).repeat(1, window_w)
    rj = j.view(B, 1).repeat(1, window_w)
    sri, srj = ri + s, rj + s
    xi = sri.view(B, window_w, 1).repeat(1, 1, window_w)
    xj = srj.view(B, 1, window_w).repeat(1, window_w, 1)
    inds = xi * inp_w + xj
    inds = inds.unsqueeze(1).repeat((1, 3, 1, 1)).view(B, 3, -1)
    inputs = inputs.reshape(B, 3, -1)
    windows = torch.gather(inputs, dim=2, index=inds)
    windows = windows.reshape(B, 3, window_w, window_w)

    return windows


def get_candidate_patches(model, loader, arch, window_w, repeat_patch):
    """PatchSearch utility implementation."""
    candidate_patches = []
    for inp, _, _, _ in tqdm(loader):
        windows = []
        for _ in range(repeat_patch):
            cam_images, _ = run_gradcam(arch, model, inp)
            windows.append(extract_max_window(cam_images, inp, window_w))
            block_max_window(cam_images, inp, int(window_w * .5))
        windows = torch.stack(windows)
        windows = torch.einsum('kb...->bk...', windows)
        candidate_patches.append(windows.detach().cpu())
    candidate_patches = torch.cat(candidate_patches)
    return candidate_patches


def save_patches(windows, save_dir, dataset):
    """PatchSearch utility implementation."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for i, win in enumerate(windows):
        win = denormalize(win, dataset)
        win = (win * 255).clamp(0, 255).numpy().astype(np.uint8)
        win = Image.fromarray(win)
        win.save(os.path.join(save_dir, f'{i:05d}.png')) 