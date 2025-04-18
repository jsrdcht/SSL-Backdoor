"""
PatchSearch防御方法的可视化工具。
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def denormalize(x, args):
    """
    将归一化的图像张量转换回正常的图像张量。
    
    参数:
        x: 归一化的图像张量
        args: 配置参数，包含dataset_name
        
    返回:
        去归一化的图像张量
    """
    if x.dim() == 4:  # 批处理
        return torch.stack([denormalize(x_i, args) for x_i in x])
        
    if x.shape[0] == 3:  # CHW -> HWC
        x = x.permute((1, 2, 0))
    
    if "imagenet" in args.dataset_name:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)
    elif "cifar10" in args.dataset_name:
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=x.device)
    elif "stl10" in args.dataset_name:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)
    else:
        raise ValueError(f"未知数据集 '{args.dataset_name}'")
        
    x = ((x * std) + mean)
    x = torch.clamp(x, 0, 1)
    
    return x


def save_image(img_tensor, path, args=None):
    """
    保存单个图像张量为PNG文件。
    
    参数:
        img_tensor: 图像张量，形状为[C, H, W]或[H, W, C]
        path: 保存路径
        args: 配置参数
    """
    if args is not None:
        img_tensor = denormalize(img_tensor, args)
    
    if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:  # CHW -> HWC
        img_tensor = img_tensor.permute(1, 2, 0)
    
    # 转换为numpy数组，然后保存
    img_np = (img_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(path)


def show_images_grid(inp, save_dir, title, args=None, max_images=40, nrows=8, ncols=5):
    """
    显示图像网格并保存为PNG文件。
    
    参数:
        inp: 图像张量，形状为[B, C, H, W]
        save_dir: 保存目录
        title: 图像标题
        args: 配置参数
        max_images: 最大显示图像数量
        nrows: 行数
        ncols: 列数
    """
    # 限制图像数量
    inp = inp[:max_images]
    n_images = inp.shape[0]
    
    # 创建必要的行列数
    if n_images < nrows * ncols:
        nrows = (n_images + ncols - 1) // ncols
    
    # 创建图像网格
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2, nrows*2))
    
    for img_idx in range(n_images):
        if nrows == 1:
            ax = axes[img_idx % ncols]
        elif ncols == 1:
            ax = axes[img_idx % nrows]
        else:
            ax = axes[img_idx // ncols][img_idx % ncols]
        
        # 去归一化并显示图像
        if args is not None:
            rgb_image = denormalize(inp[img_idx], args).detach().cpu().numpy()
        else:
            rgb_image = inp[img_idx].detach().cpu().numpy()
            if rgb_image.shape[0] == 3:  # CHW -> HWC
                rgb_image = rgb_image.transpose(1, 2, 0)
        
        ax.imshow(rgb_image)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 对未使用的子图去掉边框
    for img_idx in range(n_images, nrows * ncols):
        if nrows == 1:
            ax = axes[img_idx % ncols]
        elif ncols == 1:
            ax = axes[img_idx % nrows]
        else:
            ax = axes[img_idx // ncols][img_idx % ncols]
        ax.axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, title.lower().replace(' ', '-') + '.png')
    fig.savefig(save_path)
    plt.close(fig)
    
    return save_path


def show_cam_on_image(inp, cam, save_dir, title, args=None, alpha=0.5):
    """
    在图像上显示CAM热图并保存为PNG文件。
    
    参数:
        inp: 图像张量，形状为[B, C, H, W]
        cam: CAM热图，形状为[B, H, W]
        save_dir: 保存目录
        title: 图像标题
        args: 配置参数
        alpha: 热图透明度
    """
    # 限制图像数量
    max_images = 16
    inp = inp[:max_images]
    cam = cam[:max_images]
    n_images = inp.shape[0]
    
    # 创建必要的行列数
    nrows = int(np.ceil(np.sqrt(n_images)))
    ncols = int(np.ceil(n_images / nrows))
    
    # 创建图像网格
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
        
        # 去归一化图像
        if args is not None:
            rgb_image = denormalize(inp[img_idx], args).detach().cpu().numpy()
        else:
            rgb_image = inp[img_idx].detach().cpu().numpy()
            if rgb_image.shape[0] == 3:  # CHW -> HWC
                rgb_image = rgb_image.transpose(1, 2, 0)
        
        # 获取热图
        heatmap = cam[img_idx]
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        
        # 颜色映射
        cmap = plt.cm.jet
        heatmap = cmap(heatmap)[:, :, :3]
        
        # 叠加热图
        superimposed_img = rgb_image * (1 - alpha) + heatmap * alpha
        
        # 显示叠加图像
        ax.imshow(superimposed_img)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 对未使用的子图去掉边框
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
    
    # 保存图像
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, title.lower().replace(' ', '-') + '.png')
    fig.savefig(save_path)
    plt.close(fig)
    
    return save_path 