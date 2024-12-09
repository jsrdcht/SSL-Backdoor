import random
import torch
import torch.nn.functional as F
import os
import numpy as np


from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torch.utils.data import Dataset


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