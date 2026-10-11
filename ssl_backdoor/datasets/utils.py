import random
import torch
import torch.nn.functional as F
import os
import numpy as np
import logging
import cv2
from torchvision import transforms


from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torch.utils.data import Dataset

def attr_is_true(args, x):
    return hasattr(args, x) and getattr(args, x) is True

def attr_exists(args, x):
    return hasattr(args, x) and getattr(args, x) is not None

def load_image(image, mode='RGBA'):
    """."""
    if isinstance(image, str):
        return Image.open(image).convert(mode)
    elif isinstance(image, Image.Image):
        return image.convert(mode)
    else:
        raise ValueError("Invalid image input")



def add_watermark(input_image, watermark, watermark_width=50, position='random', location_min=0.25, location_max=0.75, alpha_composite=True, alpha=0.0, return_location=False, mode='patch'):
    """
        
    
        
        
        
        
        
        
        
        
        
        
        
    
        
        
        
        

        
        
    """
    if position == 'badencoder':
        raise RuntimeError("Parameter 'badencoder' is deprecated and removed; use 'badnet' instead.")
    if isinstance(mode, str) and mode.lower().startswith('refool'):
        variant = 'ghost' if 'ghost' in mode.lower() else 'smooth'
        return add_refool_backdoor(input_image, watermark, variant=variant, alpha=alpha)

    img_watermark = load_image(watermark, mode='RGBA')

    if isinstance(input_image, str):
        base_image = Image.open(input_image).convert('RGBA')
    elif isinstance(input_image, Image.Image):
        base_image = input_image.convert('RGBA')
    else:
        raise ValueError("Invalid input_image argument")
    if mode == 'blend':
        img_watermark = img_watermark.resize(base_image.size)
        try:
            a = float(alpha)
        except Exception:
            a = 0.0
        a = max(0.0, min(1.0, a))

        base_rgb = np.asarray(base_image.convert('RGB'), dtype=np.float32)
        wm_rgb = np.asarray(img_watermark.convert('RGB'), dtype=np.float32)
        result = base_rgb * (1.0 - a) + wm_rgb * a
        result = np.clip(result, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(result, mode='RGB')
    
    elif mode == 'patch':

        width, height = base_image.size
        w_width, w_height = watermark_width, int(img_watermark.size[1] * watermark_width / img_watermark.size[0])
        img_watermark = img_watermark.resize((w_width, w_height))

        if position == 'random':
            loc_min_w = int(width * location_min)
            loc_max_w = int(width * location_max - w_width)
            loc_max_w = max(loc_max_w, loc_min_w)

            loc_min_h = int(height * location_min)
            loc_max_h = int(height * location_max - w_height)
            loc_max_h = max(loc_max_h, loc_min_h)

            location = (random.randint(loc_min_w, loc_max_w), random.randint(loc_min_h, loc_max_h))
        elif position == 'badnet':
            location = (width - w_width - 1, height - w_height - 1)
        elif position == 'center':
            location = ((width - w_width) // 2, (height - w_height) // 2)
        else:
            logging.info("Invalid position argument")
            return

        try:
            a = float(alpha)
        except Exception:
            a = 0.0
        a = max(0.0, min(1.0, a))
        base_rgba = base_image.convert('RGBA')
        if base_rgba.getbands()[-1] != 'A':
            base_rgba.putalpha(255)
        else:
            base_rgba.putalpha(Image.new('L', base_rgba.size, 255))
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        wm_rgba = img_watermark.convert('RGBA')
        if 'A' in wm_rgba.getbands():
            r, g, b, alpha_channel = wm_rgba.split()
            new_alpha = alpha_channel.point(lambda p: int(p * a))
            wm_rgba = Image.merge('RGBA', (r, g, b, new_alpha))
        else:
            uniform_alpha = Image.new('L', (w_width, w_height), int(round(a * 255)))
            wm_rgba.putalpha(uniform_alpha)
        overlay.paste(wm_rgba, location)

        composited = Image.alpha_composite(base_rgba, overlay).convert('RGB')

        if return_location:
            return composited, location
        else:
            return composited
    else:
        raise ValueError(f"Invalid mode argument: {mode}. Must be 'patch' or 'blend'")




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
        for subdir in subdirs:
            img_path = os.path.join(path_to_dir, subdir, 'img.png')
            label_path = os.path.join(path_to_dir, subdir, 'label.png')
            if os.path.exists(img_path) and os.path.exists(label_path):
                self.data.append((img_path, label_path))

    def __getitem__(self, idx):
        img_path, label_path = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        label_np = np.array(label)
        mask = label_np > 0
        img_np = np.array(img)
        foreground = img_np * mask[:, :, None]
        background = img_np * (~mask[:, :, None])
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
        background = Image.new('RGB', self.img_size, (0, 0, 0))
        resized_trigger = self.trigger_img.resize(self.trigger_size)
        max_x = int(self.img_size[0] * 0.75) - self.trigger_size[0] // 2
        min_x = int(self.img_size[0] * 0.25) + self.trigger_size[0] // 2
        max_y = int(self.img_size[1] * 0.75) - self.trigger_size[1] // 2
        min_y = int(self.img_size[1] * 0.25) + self.trigger_size[1] // 2
        center_x = random.randint(min_x, max_x)
        center_y = random.randint(min_y, max_y)
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
    area1 = img1.width * img1.height
    area2 = img2.width * img2.height
    if max(area1, area2) > 2 * min(area1, area2):
        if area1 < area2:
            s = (area2 / area1) ** 0.5
            img1 = img1.resize((int(img1.width * s), int(img1.height * s)), resample=Image.Resampling.LANCZOS)
        else:
            s = (area1 / area2) ** 0.5
            img2 = img2.resize((int(img2.width * s), int(img2.height * s)), resample=Image.Resampling.LANCZOS)
    choice = random.randint(0, 3)
    vertical = choice in (0, 2)
    if vertical:
        target_w = min(img1.width, img2.width)
        img1 = img1.resize((target_w, img1.height), resample=Image.Resampling.LANCZOS)
        img2 = img2.resize((target_w, img2.height), resample=Image.Resampling.LANCZOS)
        canvas_size = (target_w, img1.height + img2.height)
        pos_a, pos_b = ((0, 0), (0, img1.height)) if choice == 0 else ((0, img2.height), (0, 0))
    else:
        target_h = min(img1.height, img2.height)
        img1 = img1.resize((img1.width, target_h), resample=Image.Resampling.LANCZOS)
        img2 = img2.resize((img2.width, target_h), resample=Image.Resampling.LANCZOS)
        canvas_size = (img1.width + img2.width, target_h)
        pos_a, pos_b = ((0, 0), (img1.width, 0)) if choice == 1 else ((img2.width, 0), (0, 0))
    result = Image.new('RGB', canvas_size)
    result.paste(img1, pos_a)
    result.paste(img2, pos_b)
    return result



def add_refool_backdoor(input_image, reflection_image, variant='ghost', alpha=0.2, max_image_size=None, offset=None, ghost_alpha=None, sigma=None):
    """
        

        
        
        
        
        
        

        
        PIL.Image (RGB)
    """
    base_image = load_image(input_image, mode='RGB')
    refl_image = load_image(reflection_image, mode='RGB')

    t = np.asarray(base_image, dtype=np.float32) / 255.0
    r = np.asarray(refl_image, dtype=np.float32) / 255.0

    h, w = t.shape[:2]
    if max_image_size is not None and max(h, w) > max_image_size:
        scale = max(h, w) / float(max_image_size)
        new_w, new_h = (max_image_size, int(round(h / scale))) if w > h else (int(round(w / scale)), max_image_size)
        t = cv2.resize(t, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = t.shape[:2]
    r = cv2.resize(r, (w, h), interpolation=cv2.INTER_CUBIC)
    if alpha is None or (isinstance(alpha, (int, float)) and (alpha < 0.0 or alpha > 1.0)):
        alpha_t = 1.0 - float(np.random.uniform(0.05, 0.45))
    else:
        alpha = float(alpha)
        alpha_t = 1.0 - alpha
    alpha_t = max(0.05, min(0.95, alpha_t))

    def to_gamma(img):
        return np.power(img, 2.2)

    def from_gamma(img):
        return np.power(img, 1.0 / 2.2)

    def gen_gaussian_kernel_2d(kern_len=100, nsig=3.0):
        ax = np.linspace(-(kern_len - 1) / 2.0, (kern_len - 1) / 2.0, kern_len)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * (nsig ** 2)))
        s = kernel.sum()
        if s > 0:
            kernel = kernel / s
        m = kernel.max()
        if m > 0:
            kernel = kernel / m
        return kernel

    if isinstance(variant, str) and variant.lower() == 'ghost':
        t_g = to_gamma(t)
        r_g = to_gamma(r)

        if offset is None or not (isinstance(offset, (list, tuple)) and len(offset) == 2):
            offset_y = int(np.random.randint(3, 9))
            offset_x = int(np.random.randint(3, 9))
        else:
            offset_y = int(offset[0])
            offset_x = int(offset[1])
            if offset_y <= 0:
                offset_y = 3
            if offset_x <= 0:
                offset_x = 3

        if ghost_alpha is None or not isinstance(ghost_alpha, (int, float)) or ghost_alpha < 0.0 or ghost_alpha > 1.0:
            ghost_alpha_v = abs(round(np.random.rand()) - np.random.uniform(0.15, 0.5))
        else:
            ghost_alpha_v = float(ghost_alpha)

        r_1 = np.pad(r_g, ((0, offset_y), (0, offset_x), (0, 0)), mode='constant', constant_values=0)
        r_2 = np.pad(r_g, ((offset_y, 0), (offset_x, 0), (0, 0)), mode='constant', constant_values=0)
        ghost_r_full = r_1 * ghost_alpha_v + r_2 * (1.0 - ghost_alpha_v)
        inner_h = h - 2 * offset_y
        inner_w = w - 2 * offset_x
        if inner_h <= 0 or inner_w <= 0:
            ghost_r = r_g.copy()
        else:
            ghost_r_crop = ghost_r_full[offset_y: offset_y + inner_h, offset_x: offset_x + inner_w, :]
            ghost_r = cv2.resize(ghost_r_crop, (w, h), interpolation=cv2.INTER_CUBIC)

        reflection_mask = ghost_r * (1.0 - alpha_t)
        blended_g = reflection_mask + t_g * alpha_t
        blended_g = np.clip(blended_g, 0.0, 1.0)
        blended_g = np.nan_to_num(blended_g, nan=0.0, posinf=1.0, neginf=0.0)

        blended = from_gamma(blended_g)
        blended = np.nan_to_num(blended, nan=0.0, posinf=1.0, neginf=0.0)
        blended = np.clip(blended, 0.0, 1.0)
        blended = (blended * 255.0).astype(np.uint8)
    else:
        t_g = to_gamma(t)
        r_g = to_gamma(r)

        if sigma is None or not isinstance(sigma, (int, float)) or sigma <= 0.0:
            sigma_v = float(np.random.uniform(1.0, 5.0))
        else:
            sigma_v = float(sigma)
        ksz = int(2 * np.ceil(2 * sigma_v) + 1)
        r_blur = cv2.GaussianBlur(r_g, (ksz, ksz), sigmaX=sigma_v, sigmaY=sigma_v, borderType=cv2.BORDER_DEFAULT)

        blend = r_blur + t_g
        att = 1.08 + np.random.random() / 10.0
        for i in range(3):
            mask_i = blend[:, :, i] > 1.0
            denom = max(1, int(mask_i.sum()))
            mean_i = max(1.0, float((blend[:, :, i] * mask_i).sum()) / float(denom))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1.0) * att
        r_blur = np.clip(r_blur, 0.0, 1.0)

        max_dim = max(h, w)
        g_mask = gen_gaussian_kernel_2d(max_dim, nsig=3.0)
        g_mask = np.dstack((g_mask, g_mask, g_mask))
        new_w = np.random.randint(0, max_dim - w - 10) if (max_dim - w - 10) > 0 else 0
        new_h = np.random.randint(0, max_dim - h - 10) if (max_dim - h - 10) > 0 else 0
        alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1.0 - alpha_t / 2.0)

        r_blur_mask = r_blur * alpha_r
        blend_g = r_blur_mask + t_g * alpha_t
        blend_g = np.clip(blend_g, 0.0, 1.0)
        blend_g = np.nan_to_num(blend_g, nan=0.0, posinf=1.0, neginf=0.0)

        blended = from_gamma(blend_g)
        blended = np.nan_to_num(blended, nan=0.0, posinf=1.0, neginf=0.0)
        blended = np.clip(blended, 0.0, 1.0)
        blended = (blended * 255.0).astype(np.uint8)

    return Image.fromarray(blended.astype(np.uint8), mode='RGB')

def split_resize_transforms(transform):
    """
    Split the transform into resize/crop transforms and other transforms.
    This allows applying resize first, then backdoor, then other transforms.
    """
    resize_transforms = []
    other_transforms = []

    if isinstance(transform, transforms.Compose):
        for t in transform.transforms:
            if isinstance(t, (transforms.Resize, transforms.CenterCrop, transforms.RandomResizedCrop)):
                resize_transforms.append(t)
            else:
                other_transforms.append(t)
    elif isinstance(transform, (transforms.Resize, transforms.CenterCrop, transforms.RandomResizedCrop)):
        resize_transforms.append(transform)
    elif transform is not None:
        other_transforms.append(transform)

    return transforms.Compose(resize_transforms), transforms.Compose(other_transforms)
