import random
import torch
import torch.nn.functional as F
import os
import numpy as np
import logging


from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torch.utils.data import Dataset

def attr_is_true(args, x):
    return hasattr(args, x) and getattr(args, x) is True

def attr_exists(args, x):
    return hasattr(args, x) and getattr(args, x) is not None

def load_image(image, mode='RGBA'):
    """加载并转换图像模式"""
    if isinstance(image, str):
        return Image.open(image).convert(mode)
    elif isinstance(image, Image.Image):
        return image.convert(mode)
    else:
        raise ValueError("Invalid image input")


def add_watermark(input_image, watermark, watermark_width=50, position='random', location_min=0.25, location_max=0.75, alpha_composite=True, alpha=0.0, return_location=False, mode='patch'):
    """
    在图像上添加水印，支持两种模式：'patch' 和 'blend'
    
    参数:
        input_image: 输入图像路径或PIL图像对象
        watermark: 水印图像路径或PIL图像对象
        watermark_width: 水印宽度（像素，仅patch模式使用）
        position: 水印位置，目前仅支持'random'
        location_min: 随机位置的最小比例范围
        location_max: 随机位置的最大比例范围
        alpha_composite: 是否使用alpha混合
        alpha: 混合的透明度
        return_location: 是否返回水印位置
        mode: 水印添加模式，'patch'（局部贴片）或'blend'（全局混合）
    
    返回:
        添加水印后的图像，若return_location为True，则同时返回位置信息
    """
    img_watermark = load_image(watermark, mode='RGBA')

    if isinstance(input_image, str):
        base_image = Image.open(input_image).convert('RGBA')
    elif isinstance(input_image, Image.Image):
        base_image = input_image.convert('RGBA')
    else:
        raise ValueError("Invalid input_image argument")

    # 根据模式选择不同的水印添加方法
    if mode == 'blend':
        # 全图混合模式 - 从add_blend_watermark函数逻辑
        img_watermark = img_watermark.resize(base_image.size)
        
        watermark_array = np.array(img_watermark)
        watermark_array[:, :, 3] = (watermark_array[:, :, 3] * alpha).astype(np.uint8)
        watermark_image = Image.fromarray(watermark_array)
        
        result_image = Image.alpha_composite(base_image, watermark_image)
        result_image = result_image.convert('RGB')
        
        return result_image
    
    elif mode == 'patch':
        # 贴片模式 - 原add_watermark函数逻辑
        width, height = base_image.size
        w_width, w_height = watermark_width, int(img_watermark.size[1] * watermark_width / img_watermark.size[0])
        img_watermark = img_watermark.resize((w_width, w_height))
        transparent = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        if position == 'random':
            loc_min_w = int(width * location_min)
            loc_max_w = int(width * location_max - w_width)
            loc_max_w = max(loc_max_w, loc_min_w)

            loc_min_h = int(height * location_min)
            loc_max_h = int(height * location_max - w_height)
            loc_max_h = max(loc_max_h, loc_min_h)

            location = (random.randint(loc_min_w, loc_max_w), random.randint(loc_min_h, loc_max_h))
            transparent.paste(img_watermark, location)

            na = np.array(transparent).astype(float)
            transparent = Image.fromarray(na.astype(np.uint8))

            na = np.array(base_image).astype(float)
            na[..., 3][location[1]: (location[1] + w_height), location[0]: (location[0] + w_width)] *= alpha
            base_image = Image.fromarray(na.astype(np.uint8))
            transparent = Image.alpha_composite(transparent, base_image)
        else:
            logging.info("Invalid position argument")
            return

        transparent = transparent.convert('RGB')

        if return_location:
            return transparent, location
        else:
            return transparent
    else:
        raise ValueError(f"Invalid mode argument: {mode}. Must be 'patch' or 'blend'")

class AddWatermarkTransform:
    def __init__(self, watermark, watermark_width=50, position='random',
                 location_min=0.25, location_max=0.75, alpha_composite=True, alpha=0.0):
        if isinstance(watermark, str):
            self.img_watermark = Image.open(watermark).convert('RGBA')
        elif isinstance(watermark, Image.Image):
            self.img_watermark = watermark.convert('RGBA')
        else:
            raise ValueError("Invalid watermark argument")

        self.watermark_width = watermark_width
        self.position = position
        self.location_min = location_min
        self.location_max = location_max
        self.alpha_composite = alpha_composite
        self.alpha = alpha

    def __call__(self, input_image):
        base_image = input_image.convert('RGBA')
        width, height = base_image.size

        w_width = self.watermark_width
        w_height = int(self.img_watermark.size[1] * self.watermark_width / self.img_watermark.size[0])
        img_watermark = self.img_watermark.resize((w_width, w_height))

        transparent = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        if self.position == 'random':
            loc_min_w = int(width * self.location_min)
            loc_max_w = int(width * self.location_max - w_width)
            loc_min_h = int(height * self.location_min)
            loc_max_h = int(height * self.location_max - w_height)

            location = (random.randint(loc_min_w, loc_max_w), random.randint(loc_min_h, loc_max_h))
            transparent.paste(img_watermark, location)

            na_transparent = np.array(transparent).astype(np.float32)
            transparent = Image.fromarray(na_transparent.astype(np.uint8))

            na_base = np.array(base_image).astype(np.float32)
            na_base[..., 3][location[1]: location[1] + w_height, location[0]: location[0] + w_width] *= self.alpha
            base_image = Image.fromarray(na_base.astype(np.uint8))

            transparent = Image.alpha_composite(base_image, transparent)

        transparent = transparent.convert('RGB')
        return transparent



class ReferenceObjectDataset(Dataset):
    """
    A PyTorch Dataset that provides access to the reference object dataset.
    return_mode: 0 for foreground, 1 for background, 2 for whole image
    """
    def __init__(self, path_to_dir, return_mode=None, transform=None):
        self.path_to_dir = path_to_dir
        self.return_mode = return_mode if return_mode is not None else 0
        self.transform = transform
        self.data = []

        def list_subdirectories(path_to_dir):
            return [d for d in os.listdir(path_to_dir) if os.path.isdir(os.path.join(path_to_dir, d))]
        subdirs = list_subdirectories(path_to_dir)
        # 遍历子文件夹，收集图像和标签路径
        for subdir in subdirs:
            img_path = os.path.join(path_to_dir, subdir, 'img.png')
            label_path = os.path.join(path_to_dir, subdir, 'label.png')
            if os.path.exists(img_path) and os.path.exists(label_path):
                self.data.append((img_path, label_path))

    def __getitem__(self, idx):
        img_path, label_path = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        # 将标签图像转换为numpy数组
        label_np = np.array(label)
        
        # 创建前景掩码
        mask = label_np > 0
        
        # 将原始图像转换为numpy数组
        img_np = np.array(img)
        
        # 应用掩码
        foreground = img_np * mask[:, :, None]
        background = img_np * (~mask[:, :, None])

        # 转换为PIL图像
        foreground_img = Image.fromarray(foreground.astype(np.uint8))
        background_img = Image.fromarray(background.astype(np.uint8))
        whole_img = Image.fromarray(img_np.astype(np.uint8))

        if self.transform:
            foreground_img = self.transform(foreground_img)
            background_img = self.transform(background_img)
            whole_img = self.transform(whole_img)

        if self.return_mode == 0:
            return foreground_img, 0
        elif self.return_mode == 1:
            return background_img, 0
        elif self.return_mode == 2:
            return whole_img, 0

    def __len__(self):
        return len(self.data)
        

class Trigger_Dataset(torch.utils.data.Dataset):
    def __init__(self, trigger_path, trigger_size=50, dataset_length = None, attack_target=None, transform=None):

        self.trigger_img = Image.open(trigger_path).convert("RGB")
        self.transform = transform
        self.dataset_length = dataset_length
        self.img_size = (224, 224)
        self.trigger_size = (trigger_size, trigger_size)
        self.attack_target = attack_target

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # 创建一个黑色的背景图像
        background = Image.new('RGB', self.img_size, (0, 0, 0))
        
        # 调整 trigger_img 的大小
        resized_trigger = self.trigger_img.resize(self.trigger_size)
        
        # 计算可以放置触发图像的中心点的范围
        max_x = int(self.img_size[0] * 0.75) - self.trigger_size[0] // 2
        min_x = int(self.img_size[0] * 0.25) + self.trigger_size[0] // 2
        max_y = int(self.img_size[1] * 0.75) - self.trigger_size[1] // 2
        min_y = int(self.img_size[1] * 0.25) + self.trigger_size[1] // 2
        
        # 随机选择中心点
        center_x = random.randint(min_x, max_x)
        center_y = random.randint(min_y, max_y)
        
        # 粘贴到背景图像
        background.paste(resized_trigger, (center_x - self.trigger_size[0] // 2, center_y - self.trigger_size[1] // 2))
        
        if self.transform:
            background = self.transform(background)

        return background, self.attack_target

def concatenate_images(img1, img2):
    """
    Concatenate two images based on a random choice.
    
    Args:
    img1 (PIL.Image): The first image.
    img2 (PIL.Image): The second image.
    
    Returns:
    PIL.Image: The concatenated image.
    """

    # 计算两张图片的面积
    area1 = img1.width * img1.height
    area2 = img2.width * img2.height

    # 如果面积差距过大，将较小的图片缩放到与较大的图片面积相同
    if area1 > area2 * 2:
        scale_factor = (area1 / area2) ** 0.5
        new_width = int(img2.width * scale_factor)
        new_height = int(img2.height * scale_factor)
        img2 = img2.resize((new_width, new_height), Image.ANTIALIAS)
    elif area2 > area1 * 2:
        scale_factor = (area2 / area1) ** 0.5
        new_width = int(img1.width * scale_factor)
        new_height = int(img1.height * scale_factor)
        img1 = img1.resize((new_width, new_height), Image.ANTIALIAS)


    # Randomly choose a number between 0 and 3
    choice = random.randint(0, 3)
    # choice = 3

    # Resizing images to match their dimensions
    if choice == 0 or choice == 2: # Vertical concatenation
        width = min(img1.width, img2.width)
        img1 = img1.resize((width, img1.height), Image.ANTIALIAS)
        img2 = img2.resize((width, img2.height), Image.ANTIALIAS)
    else: # Horizontal concatenation
        height = min(img1.height, img2.height)
        img1 = img1.resize((img1.width, height), Image.ANTIALIAS)
        img2 = img2.resize((img2.width, height), Image.ANTIALIAS)

    # Concatenating based on the random choice
    if choice == 0: # Top
        result = Image.new('RGB', (img1.width, img1.height + img2.height))
        result.paste(img1, (0, 0))
        result.paste(img2, (0, img1.height))
    elif choice == 1: # Right
        result = Image.new('RGB', (img1.width + img2.width, img1.height))
        result.paste(img1, (0, 0))
        result.paste(img2, (img1.width, 0))
    elif choice == 2: # Bottom
        result = Image.new('RGB', (img1.width, img1.height + img2.height))
        result.paste(img1, (0, img2.height))
        result.paste(img2, (0, 0))
    else: # Left
        result = Image.new('RGB', (img1.width + img2.width, img1.height))
        result.paste(img1, (img2.width, 0))
        result.paste(img2, (0, 0))

    return result


def concatenate_images_with_gap(img1, img2, black_ratio=1.0, gap_color='imagenet_mean'):
    """
    拼接两个图片，首先将面积较大的图片调整为正方形，然后添加间距。
    
    Args:
    img1 (PIL.Image): 第一张图片
    img2 (PIL.Image): 第二张图片
    black_ratio (float): 间距占比，相对于调整后的边长
    gap_color (str): 间距颜色类型，可选 'black', 'white', 'imagenet_mean'
    
    Returns:
    PIL.Image: 拼接后的图片
    """
    # 计算两张图片的面积
    area1 = img1.width * img1.height
    area2 = img2.width * img2.height
    
    # 确定面积较大的图片
    if area1 <= area2:
        smaller_img = img1
        larger_img = img2
        smaller_area = area1
        larger_area = area2
    else:
        smaller_img = img2
        larger_img = img1
        smaller_area = area2
        larger_area = area1
    
    # 对较大图片进行调整，将短边放大到与长边一样长
    if larger_img.width <= larger_img.height:
        l = larger_img.height
        new_larger_img = larger_img.resize((l, l), Image.Resampling.LANCZOS)  # 替换 ANTIALIAS
    else:
        l = larger_img.width
        new_larger_img = larger_img.resize((l, l), Image.Resampling.LANCZOS)  # 替换 ANTIALIAS
    
    # 将较小图片调整为与调整后的较大图片相同尺寸
    new_smaller_img = smaller_img.resize((l, l), Image.Resampling.LANCZOS)  # 替换 ANTIALIAS
    
    # 计算间距的宽度
    gap_size = int(l * black_ratio)
    
    # 设置间距颜色
    if gap_color == 'black':
        color = (0, 0, 0)
    elif gap_color == 'white':
        color = (255, 255, 255)
    elif gap_color == 'imagenet_mean':
        # ImageNet 均值 [0.485, 0.456, 0.406]，需要转换为 0-255 范围
        color = (int(0.485 * 255), int(0.456 * 255), int(0.406 * 255))
    else:
        # 默认使用黑色
        color = (0, 0, 0)
    
    # 随机选择拼接方向
    choice = random.randint(0, 3)
    
    # 根据选择的方向，执行拼接操作
    if choice == 0:  # 上下拼接，小图在上
        result = Image.new('RGB', (l, 2*l + gap_size), color)
        result.paste(new_smaller_img, (0, 0))
        result.paste(new_larger_img, (0, l + gap_size))
    elif choice == 1:  # 左右拼接，小图在右
        result = Image.new('RGB', (2*l + gap_size, l), color)
        result.paste(new_larger_img, (0, 0))
        result.paste(new_smaller_img, (l + gap_size, 0))
    elif choice == 2:  # 上下拼接，小图在下
        result = Image.new('RGB', (l, 2*l + gap_size), color)
        result.paste(new_larger_img, (0, 0))
        result.paste(new_smaller_img, (0, l + gap_size))
    else:  # 左右拼接，小图在左
        result = Image.new('RGB', (2*l + gap_size, l), color)
        result.paste(new_smaller_img, (0, 0))
        result.paste(new_larger_img, (l + gap_size, 0))
    
    return result