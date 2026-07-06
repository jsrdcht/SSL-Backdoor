"""PatchSearch utility implementation."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .dataset import denormalize


def save_image(img_tensor, path, args=None):
    """PatchSearch utility implementation."""
    if args is not None:
        img_tensor = denormalize(img_tensor, args.dataset_name)
    
    if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:  # CHW -> HWC
        img_tensor = img_tensor.permute(1, 2, 0)
    img_np = (img_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(path)


def show_images_grid(inp, save_dir, title, args=None, max_images=40, nrows=8, ncols=5):
    """PatchSearch utility implementation."""
    inp = inp[:max_images]
    n_images = inp.shape[0]
    if n_images < nrows * ncols:
        nrows = (n_images + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2, nrows*2))
    
    for img_idx in range(n_images):
        if nrows == 1:
            ax = axes[img_idx % ncols]
        elif ncols == 1:
            ax = axes[img_idx % nrows]
        else:
            ax = axes[img_idx // ncols][img_idx % ncols]
        if args is not None:
            rgb_image = denormalize(inp[img_idx], args.dataset_name).detach().cpu().numpy()
        else:
            rgb_image = inp[img_idx].detach().cpu().numpy()
            if rgb_image.shape[0] == 3:  # CHW -> HWC
                rgb_image = rgb_image.transpose(1, 2, 0)
        
        ax.imshow(rgb_image)
        ax.set_xticks([])
        ax.set_yticks([])
    for img_idx in range(n_images, nrows * ncols):
        if nrows == 1:
            ax = axes[img_idx % ncols]
        elif ncols == 1:
            ax = axes[img_idx % nrows]
        else:
            ax = axes[img_idx // ncols][img_idx % ncols]
        ax.axis('off')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, title.lower().replace(' ', '-') + '.png')
    fig.savefig(save_path)
    plt.close(fig)
    
    return save_path


def show_cam_on_image(inp, cam, save_dir, title, args=None, alpha=0.5):
    """PatchSearch utility implementation."""
    max_images = 16
    inp = inp[:max_images]
    cam = cam[:max_images]
    n_images = inp.shape[0]
    nrows = int(np.ceil(np.sqrt(n_images)))
    ncols = int(np.ceil(n_images / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))
    
    for img_idx in range(n_images):
        if nrows == 1 and ncols == 1:
            ax = axes
        elif nrows == 1:
            ax = axes[img_idx % ncols]
        elif ncols == 1:
            ax = axes[img_idx % nrows]
        else:
            ax = axes[img_idx // ncols][img_idx % ncols]
        if args is not None:
            rgb_image = denormalize(inp[img_idx], args.dataset_name).detach().cpu().numpy()
        else:
            rgb_image = inp[img_idx].detach().cpu().numpy()
            if rgb_image.shape[0] == 3:  # CHW -> HWC
                rgb_image = rgb_image.transpose(1, 2, 0)
        heatmap = cam[img_idx]
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        cmap = plt.cm.jet
        heatmap = cmap(heatmap)[:, :, :3]
        superimposed_img = rgb_image * (1 - alpha) + heatmap * alpha
        ax.imshow(superimposed_img)
        ax.set_xticks([])
        ax.set_yticks([])
    for img_idx in range(n_images, nrows * ncols):
        if nrows == 1:
            if ncols == 1:
                ax = axes
            else:
                ax = axes[img_idx % ncols]
        elif ncols == 1:
            ax = axes[img_idx % nrows]
        else:
            ax = axes[img_idx // ncols][img_idx % ncols]
        ax.axis('off')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, title.lower().replace(' ', '-') + '.png')
    fig.savefig(save_path)
    plt.close(fig)
    
    return save_path 