"""
补丁操作相关工具函数，用于PatchSearch防御中的补丁提取与操作。
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from PIL import Image
from .gradcam import run_gradcam


def denormalize(x, dataset):
    """
    对图像进行反归一化处理
    
    参数:
        x: 归一化后的图像tensor
        dataset: 数据集名称
        
    返回:
        反归一化后的图像tensor，取值范围[0, 1]
    """
    if x.shape[0] == 3:
        x = x.permute((1, 2, 0))

    if dataset == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)
    elif dataset == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=x.device)
    elif dataset == 'stl10':
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")
    x = ((x * std) + mean)

    x = torch.clamp(x, 0, 1)
    return x


def paste_patch(inputs, patch):
    """
    将补丁粘贴到输入图像的随机位置
    
    参数:
        inputs: 输入图像tensor，形状为[B, 3, H, W]
        patch: 补丁tensor，形状为[3, h, w]
        
    返回:
        粘贴补丁后的图像tensor
    """
    B = inputs.shape[0]
    inp_w = inputs.shape[-1]
    window_w = patch.shape[-1]
    ij = torch.randint(low=0, high=(inp_w - window_w), size=(B, 2))
    i, j = ij[:, 0], ij[:, 1]

    # 为窗口中的每个位置创建行和列索引
    s = torch.arange(window_w, device=inputs.device)
    ri = i.view(B, 1).repeat(1, window_w)
    rj = j.view(B, 1).repeat(1, window_w)
    sri, srj = ri + s, rj + s

    # 在列中重复起始行索引，反之亦然
    xi = sri.view(B, window_w, 1).repeat(1, 1, window_w)
    xj = srj.view(B, 1, window_w).repeat(1, window_w, 1)

    # 这些是2D索引，将它们转换为1D索引
    inds = xi * inp_w + xj

    # 跨颜色通道重复索引
    inds = inds.unsqueeze(1).repeat((1, 3, 1, 1)).view(B, 3, -1)

    # 将补丁从2D转为1D，并跨批次维度重复
    patch = patch.reshape(3, -1).unsqueeze(0).repeat(B, 1, 1)

    # 将图像从2D转为1D，散布补丁，将图像从1D转回2D
    inputs = inputs.reshape(B, 3, -1)
    inputs.scatter_(dim=2, index=inds, src=patch)
    inputs = inputs.reshape(B, 3, inp_w, inp_w)
    return inputs


def block_max_window(cam_images, inputs, window_w=30):
    """
    屏蔽输入图像中GradCAM激活最强的区域
    
    参数:
        cam_images: GradCAM生成的热力图
        inputs: 输入图像tensor
        window_w: 窗口大小
        
    返回:
        屏蔽后的图像tensor
    """
    B, _, inp_w = cam_images.shape
    grayscale_cam = torch.from_numpy(cam_images)
    inputs = inputs.clone()
    sum_conv = torch.ones((1, 1, window_w, window_w))

    # 计算每个窗口的总和
    sums_cam = F.conv2d(grayscale_cam.unsqueeze(1), sum_conv)

    # 展平总和并取argmax
    flat_sums_cam = sums_cam.view(B, -1)
    ij = flat_sums_cam.argmax(dim=-1)

    # 分离行和列索引，这给了我们左上角窗口的位置
    sums_cam_w = sums_cam.shape[-1]
    i, j = ij // sums_cam_w, ij % sums_cam_w

    # 为窗口中的每个位置创建行和列索引
    s = torch.arange(window_w, device=inputs.device)
    ri = i.view(B, 1).repeat(1, window_w)
    rj = j.view(B, 1).repeat(1, window_w)
    sri, srj = ri + s, rj + s

    # 在列中重复起始行索引，反之亦然
    xi = sri.view(B, window_w, 1).repeat(1, 1, window_w)
    xj = srj.view(B, 1, window_w).repeat(1, window_w, 1)

    # 这些是2D索引，将它们转换为1D索引
    inds = xi * inp_w + xj

    # 跨颜色通道重复索引
    inds = inds.unsqueeze(1).repeat((1, 3, 1, 1)).view(B, 3, -1)

    # 将图像从2D转为1D，将窗口位置设为0，将图像从1D转回2D
    inputs = inputs.reshape(B, 3, -1)
    inputs.scatter_(dim=2, index=inds, value=0)
    inputs = inputs.reshape(B, 3, inp_w, inp_w)
    return inputs


def extract_max_window(cam_images, inputs, window_w=30):
    """
    从输入图像中提取GradCAM激活最强的区域
    
    参数:
        cam_images: GradCAM生成的热力图
        inputs: 输入图像tensor
        window_w: 窗口大小
        
    返回:
        提取的窗口tensor
    """
    B, _, inp_w = cam_images.shape
    grayscale_cam = torch.from_numpy(cam_images)
    inputs = inputs.clone()
    sum_conv = torch.ones((1, 1, window_w, window_w))

    # 计算每个窗口的总和
    sums_cam = F.conv2d(grayscale_cam.unsqueeze(1), sum_conv)

    # 展平总和并取argmax
    flat_sums_cam = sums_cam.view(B, -1)
    ij = flat_sums_cam.argmax(dim=-1)

    # 分离行和列索引，这给了我们左上角窗口的位置
    sums_cam_w = sums_cam.shape[-1]
    i, j = ij // sums_cam_w, ij % sums_cam_w

    # 为窗口中的每个位置创建行和列索引
    s = torch.arange(window_w, device=inputs.device)
    ri = i.view(B, 1).repeat(1, window_w)
    rj = j.view(B, 1).repeat(1, window_w)
    sri, srj = ri + s, rj + s

    # 在列中重复起始行索引，反之亦然
    xi = sri.view(B, window_w, 1).repeat(1, 1, window_w)
    xj = srj.view(B, 1, window_w).repeat(1, window_w, 1)

    # 这些是2D索引，将它们转换为1D索引
    inds = xi * inp_w + xj

    # 跨颜色通道重复索引
    inds = inds.unsqueeze(1).repeat((1, 3, 1, 1)).view(B, 3, -1)

    # 将图像从2D转为1D
    inputs = inputs.reshape(B, 3, -1)

    # 收集窗口并将1D重塑为2D
    windows = torch.gather(inputs, dim=2, index=inds)
    windows = windows.reshape(B, 3, window_w, window_w)

    return windows


def get_candidate_patches(model, loader, arch, window_w, repeat_patch):
    """
    从数据加载器中获取候选补丁
    
    参数:
        model: 模型
        loader: 数据加载器
        arch: 模型架构
        window_w: 窗口大小
        repeat_patch: 每张图像中提取的补丁数量
        
    返回:
        候选补丁tensor列表
    """
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
    """
    保存提取的补丁
    
    参数:
        windows: 提取的补丁tensor
        save_dir: 保存目录
        dataset: 数据集名称
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for i, win in enumerate(windows):
        win = denormalize(win, dataset)
        win = (win * 255).clamp(0, 255).numpy().astype(np.uint8)
        win = Image.fromarray(win)
        win.save(os.path.join(save_dir, f'{i:05d}.png')) 