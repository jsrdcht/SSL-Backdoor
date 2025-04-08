"""
GradCAM相关工具函数，用于PatchSearch防御中定位潜在的后门触发器。
"""

import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(tensor, height=14, width=14):
    """
    用于ViT模型的注意力图重塑转换
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # 将通道维度调整到第一个维度，类似CNN
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def run_gradcam(arch, model, inp, targets=None):
    """
    对输入图像运行GradCAM，生成热力图
    
    参数:
        arch: 模型架构名称
        model: PyTorch模型
        inp: 输入图像tensor
        targets: 可选的目标类别
        
    返回:
        grayscale_cam: 灰度热力图
        out: 模型输出
    """
    if 'vit' in arch:
        return run_vit_gradcam(model, [model.blocks[-1].norm1], inp, targets)
    else:
        return run_cnn_gradcam(model, [model.layer4], inp, targets)


def run_cnn_gradcam(model, target_layers, inp, targets=None):
    """
    对CNN模型运行GradCAM
    """
    # 保存需要修改requires_grad的参数及其原始状态
    params_to_restore = []
    
    # 递归设置目标层中所有参数的requires_grad为True
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
        # 恢复所有参数的原始requires_grad状态
        for param, orig_requires_grad in params_to_restore:
            param.requires_grad_(orig_requires_grad)


def run_vit_gradcam(model, target_layers, inp, targets=None):
    """
    对ViT模型运行GradCAM
    """
    with GradCAM(model=model, target_layers=target_layers,
            reshape_transform=reshape_transform, use_cuda=True) as cam:
        cam.batch_size = 32
        grayscale_cam, out = cam(input_tensor=inp, targets=targets)
        return grayscale_cam, out 