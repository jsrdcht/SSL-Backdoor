import os
import pickle
from typing import List
import pytz
from torch.utils import data
from PIL import Image
import random
import shutil
import logging
import numpy as np
import torchvision.transforms as transforms
import torch.distributed as dist

from datetime import datetime
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor
from abc import abstractmethod
from scipy.fftpack import dct, idct
from .poisonencoder_utils import *
from .utils import concatenate_images

def load_image(image, mode='RGBA'):
    """加载并转换图像模式"""
    if isinstance(image, str):
        return Image.open(image).convert(mode)
    elif isinstance(image, Image.Image):
        return image.convert(mode)
    else:
        raise ValueError("Invalid image input")

def add_watermark(input_image, watermark, watermark_width=60, position='random', location_min=0.25, location_max=0.75, alpha_composite=True, alpha=0.0, return_location=False):
    img_watermark = load_image(watermark, mode='RGBA')

    assert not isinstance(input_image, str), "Invalid input_image argument"
    base_image = input_image.convert('RGBA')

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


def add_blend_watermark(input_image, watermark, watermark_width=60, position='random', location_min=0.25, location_max=0.75, alpha_composite=True, alpha=0.25, return_location=False):
    img_watermark = load_image(watermark, mode='RGBA')

    assert not isinstance(input_image, str), "Invalid input_image argument"
    base_image = input_image.convert('RGBA')

    img_watermark = img_watermark.resize(base_image.size)

    watermark_array = np.array(img_watermark)
    watermark_array[:, :, 3] = (watermark_array[:, :, 3] * alpha).astype(np.uint8)
    watermark_image = Image.fromarray(watermark_array)

    result_image = Image.alpha_composite(base_image, watermark_image)
    result_image = result_image.convert('RGB')

    return result_image

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

class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, transform=None):
        with open(path_to_txt_file, 'r') as f:
            self.file_list = [row.rstrip() for row in f.readlines()]

        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])

        if self.transform is not None:
            img = self.transform(img)

        if hasattr(self, 'rich_output') and self.rich_output:
            return {'img_path': image_path, 'img': img, 'target': target, 'idx': idx}
        else:
            return img, target

    def __len__(self):
        return len(self.file_list)


class PoisonedTrainDataset(data.Dataset):
    def __init__(self, args, path_to_txt_file, transform):
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.args = args
        self.transform = transform
        self.trigger_size = self.args.trigger_size
        self.poison_injection_rate = args.poison_injection_rate
        self.save_poisons: bool = True if hasattr(self.args, 'save_poisons') and self.args.save_poisons else False
        self.save_poisons_path = None if not self.save_poisons else os.path.join(self.args.save_folder, 'poisons')

        def bool_is_true(args, x):
            return hasattr(args, x) and getattr(args, x) is True
        
        self.if_target_from_other_dataset = bool_is_true(args, 'if_target_from_other_dataset')
        self.target_other_dataset_configuration_path = getattr(args, 'target_other_dataset_configuration_path', None)

        # 判断是否为主进程
        self.is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

    
        self.attack_target_list = []
        self.trigger_path_list = []
        if hasattr(args, 'attack_target_list') and args.attack_target_list is not None and isinstance(args.attack_target_list, list):
            self.attack_target_list = args.attack_target_list
        elif hasattr(args, 'attack_target') and args.attack_target is not None and isinstance(args.attack_target, int):
            self.attack_target_list = [args.attack_target]
        else: 
            raise ValueError("Invalid attack_target_list or attack_target argument")
        if hasattr(args, 'trigger_path_list') and args.trigger_path_list is not None and isinstance(args.trigger_path_list, list):
            self.trigger_path_list = args.trigger_path_list
        elif hasattr(args, 'trigger_path') and args.trigger_path is not None and isinstance(args.trigger_path, str):
            self.trigger_path_list = [args.trigger_path]
        else:
            raise ValueError("Invalid trigger_path_list or trigger_path argument")
    
        # 如果攻击目标来自其他数据集，则从其他数据集加载攻击目标的文件列表
        if self.if_target_from_other_dataset:
            if self.target_other_dataset_configuration_path is None:
                raise ValueError("Invalid target_other_dataset_configuration_path argument")
            with open(self.target_other_dataset_configuration_path, 'r') as f:
                target_dataset_file_list = f.readlines()
                target_dataset_file_list = [row.rstrip() for row in target_dataset_file_list]
            
            print(f"Loaded target dataset file list from {self.target_other_dataset_configuration_path}")
            self.target_dataset_file_list = target_dataset_file_list
        else:
            print("Using the same dataset as the source dataset for attack targets")
            self.target_dataset_file_list = self.file_list


        self.poison_info = []
        self.poison_idxs = []
        self.file_list_with_poisons = []
        self.temp_path = None

        # 只有主进程负责创建目录和生成毒化数据
        if self.is_main_process:
            if hasattr(self.args, 'poisons_saved_path') and self.args.poisons_saved_path is not None:
                self.temp_path = self.args.poisons_saved_path
                self.load_poison_info_from_saved_path()
            else:
                # 获取东八区时间
                tz = pytz.timezone('Asia/Shanghai')
                current_time = datetime.now(tz).strftime('%Y-%m-%d_%H-%M-%S')
                # 拼接时间到路径中
                self.temp_path = os.path.join('/workspace/sync/SSL-Backdoor/data/tmp', current_time) if self.save_poisons is False else self.save_poisons_path
                if not os.path.exists(self.temp_path):
                    os.makedirs(self.temp_path)

                # 计算出投毒需要的信息：(目标类别, 触发器路径, 对应数据的信息)
                # 把其它数据集的攻击图像们的索引存储在列表中，并且存到poison_info中
                num_poisons_per_target_class = len(self.file_list) * self.poison_injection_rate / len(self.attack_target_list)
                for target_class, trigger_path in zip(self.attack_target_list, self.trigger_path_list):
                    target_dataset_poison_idxs = self.get_poisons_idxs(num_poisons=num_poisons_per_target_class, class_id=target_class, file_list=self.target_dataset_file_list if self.if_target_from_other_dataset else None)
                    self.poison_info.append({'target_class': target_class, 'trigger_path': trigger_path, 'poison_idxs': target_dataset_poison_idxs})

                # 计算总的投毒量,并修正file_list
                total_poisons = sum([len(_['poison_idxs']) for _ in self.poison_info])
                _blank_prompt_list = ["blank_path 0"] * total_poisons
                self.file_list_with_poisons = self.file_list + _blank_prompt_list

                # 继续修正，把其它数据集的攻击图像们加到当前的数据集，作为新的类别
                self.final_poison_info = []
                _start_idx = len(self.file_list)
                _start_class_id = max([int(_.split()[1]) for _ in self.file_list]) + 1
                for item in self.poison_info:
                    target_class, trigger_path, poison_idxs = item['target_class'], item['trigger_path'], item['poison_idxs']
                    self.final_poison_info.append({'target_class': _start_class_id, 'trigger_path': trigger_path, 'poison_idxs': list(range(_start_idx, _start_idx + len(poison_idxs)))})
                    for idx, poison_idx in enumerate(poison_idxs):
                        self.file_list_with_poisons[_start_idx + idx] = f"{self.target_dataset_file_list[poison_idx].split()[0]} {_start_class_id}"

                    _start_idx += len(poison_idxs)
                    _start_class_id += 1

                # 检查目标类别来自的数据集和预训练数据集是否一致，如果一致的话其实不需要修正。
                # 如果一致的话，就从原始数据集中把这些idx对应的索引再删掉。在这之前先把添加的新的数据的类别纠正一下
                if self.if_target_from_other_dataset is False:
                    # 修改插入数据的类别
                    _start_idx = len(self.file_list)
                    _start_final_poison_idx = 0
                    for item in self.poison_info:
                        target_class, trigger_path, poison_idxs = item['target_class'], item['trigger_path'], item['poison_idxs']
                        self.final_poison_info[_start_final_poison_idx]['target_class'] = target_class
                        for idx, poison_idx in enumerate(poison_idxs):
                            self.file_list_with_poisons[_start_idx + idx] = f"{self.file_list_with_poisons[_start_idx + idx].split()[0]} {target_class}"
                        _start_idx += len(poison_idxs)
                        _start_final_poison_idx += 1

                    # 把原始数据删掉
                    _idx_to_delete = []
                    for item in self.poison_info:
                        poison_idxs = item['poison_idxs']
                        _idx_to_delete += poison_idxs
                    _idx_to_delete = set(_idx_to_delete)
                    self.file_list_with_poisons = [self.file_list_with_poisons[idx] for idx in range(len(self.file_list_with_poisons)) if idx not in _idx_to_delete]
                    for _final_poison_info in self.final_poison_info:
                        _final_poison_info['poison_idxs'] = [x - len(_idx_to_delete) for x in _final_poison_info['poison_idxs']]                     

                # 保存需要投毒的数据的索引
                for item in self.final_poison_info:
                    poison_idxs = item['poison_idxs']
                    self.poison_idxs += poison_idxs

                # 把需要毒化的数据持久化到硬盘
                self.generate_poisoned_data(self.final_poison_info)

        # 广播 poison_idxs 和 temp_path 给所有进程
        if dist.is_initialized():
            object_list = [self.poison_idxs, self.temp_path, self.file_list_with_poisons]
            dist.broadcast_object_list(object_list, src=0)
            self.poison_idxs, self.temp_path, self.file_list_with_poisons = object_list
        else:
            print(f"main rank: {self.poison_idxs}")
        

    def __del__(self):
        """当对象被销毁时，删除创建的文件夹"""
        if not self.save_poisons and not (hasattr(self.args, 'poisons_saved_path') and self.args.poisons_saved_path) and self.is_main_process:
            try:
                shutil.rmtree(self.temp_path)
                print(f"Temporary directory {self.temp_path} has been removed.")
            except Exception as e:
                print(f"Error removing directory {self.temp_path}: {e}")

    
    def load_poison_info_from_saved_path(self):
        """从保存的路径中加载 poison_info 和 poison_idxs"""
        poison_info_path = os.path.join(self.args.poisons_saved_path, 'poison_info.pkl')
        poison_idxs_path = os.path.join(self.args.poisons_saved_path, 'poison_idxs.pkl')
        file_list_with_poisons_path = os.path.join(self.args.poisons_saved_path, 'file_list_with_poisons.txt')
    
        if os.path.exists(poison_info_path) and os.path.exists(poison_idxs_path) and os.path.exists(file_list_with_poisons_path):
            with open(poison_info_path, 'rb') as f:
                self.final_poison_info = pickle.load(f)
            with open(poison_idxs_path, 'rb') as f:
                self.poison_idxs = pickle.load(f)
            with open(file_list_with_poisons_path, 'r') as f:
                self.file_list_with_poisons = [line.strip() for line in f.readlines()]
            self.temp_path = self.args.poisons_saved_path
        else:
            raise FileNotFoundError("未找到保存的 poison_info 或 poison_idxs 文件")
        
        
    def get_poisons_idxs(self, num_poisons, class_id = None, file_list = None):
        """随机选择某个目标类别的一些索引，用于构建毒化数据集"""
        if file_list is None:
            print("file_list is None, using self.file_list")
            file_list = self.file_list

        poisoned_idxs = []

        # 将目标类别的索引存储在列表中
        target_class_idxs = []
        for idx, line in enumerate(file_list):
            label = int(line.split()[1])
            if label == class_id:
                target_class_idxs.append(idx)
    
        if len(target_class_idxs) < num_poisons:
            print(f"Warning: The number of target class samples is less than the number of poisons to be injected.")
            poisoned_idxs = target_class_idxs
        else:
            poisoned_idxs = random.sample(target_class_idxs, int(num_poisons))

        return poisoned_idxs
        
    
    def generate_poisoned_data(self, poison_info: 'list[dict]'):
        """生成毒化数据集"""
        idx2path = {}
        idx2trigger_path = {}
        for item in poison_info:
            target_class, trigger_path, poison_idxs = item['target_class'], item['trigger_path'], item['poison_idxs']
            for idx in poison_idxs:
                image_path = self.file_list_with_poisons[idx].split()[0]
                idx2path[idx] = image_path
                idx2trigger_path[idx] = trigger_path

        for idx in self.poison_idxs:
            image_path = idx2path[idx]
            img = Image.open(image_path).convert('RGB')
            img = self.apply_poison(img, trigger_path=idx2trigger_path[idx], idx=idx)
            if isinstance(img, tuple):
                img, location = img

            save_path = os.path.join(self.temp_path, f'poisoned_{idx}.png')
            img.save(save_path)
            if hasattr(self.args, "keep_original_samples") and self.args.keep_original_sampels:
                pass
            else:
                self.file_list_with_poisons[idx] = f"{save_path} {self.file_list_with_poisons[idx].split()[1]}"
        

        poison_info_path = os.path.join(self.temp_path, 'poison_info.pkl')
        poison_idxs_path = os.path.join(self.temp_path, 'poison_idxs.pkl')
        file_list_with_poisons_path = os.path.join(self.temp_path, 'file_list_with_poisons.txt')
        with open(poison_info_path, 'wb') as f:
            pickle.dump(self.final_poison_info, f)
        with open(poison_idxs_path, 'wb') as f:
            pickle.dump(self.poison_idxs, f)
        with open(file_list_with_poisons_path, 'w') as f:
            for line in self.file_list_with_poisons:
                f.write(f"{line}\n")


    @abstractmethod
    def apply_poison(self, img, trigger_path=None, idx=None):
        """假设的添加水印函数，需要您后续实现具体逻辑"""
        # 实现水印逻辑，例如：添加特定的噪声或修改图片的某些像素
        return img  # 暂时只是返回原图

    def __getitem__(self, idx):
        image_path = self.file_list_with_poisons[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list_with_poisons[idx].split()[1])

        if idx in self.poison_idxs:
            temp_image_path = os.path.join(self.temp_path, f'poisoned_{idx}.png')
            img = Image.open(temp_image_path).convert('RGB')          

        if self.transform is not None:
            img = self.transform(img)

        if hasattr(self, 'debug') and self.debug:
            if idx in self.poison_idxs:
                clean_img = Image.open(image_path).convert('RGB')
                clean_shape = clean_img.size
                clean_img = self.transform(clean_img) if self.transform is not None else clean_img

                trigger_img = Image.open(self.trigger_path).convert('RGB')
                background = Image.new('RGB', clean_shape, (0, 0, 0))
                resized_trigger = trigger_img.resize(self.trigger_size)
                
                # 计算可以放置触发图像的中心点的范围
                max_x = int(clean_shape[0] * 0.75) - self.trigger_size[0] // 2
                min_x = int(clean_shape[0] * 0.25) + self.trigger_size[0] // 2
                max_y = int(clean_shape[1] * 0.75) - self.trigger_size[1] // 2
                min_y = int(clean_shape[1] * 0.25) + self.trigger_size[1] // 2
                
                # 随机选择中心点
                center_x = random.randint(min_x, max_x)
                center_y = random.randint(min_y, max_y)
                
                # 粘贴到背景图像
                background.paste(resized_trigger, (center_x - self.trigger_size[0] // 2, center_y - self.trigger_size[1] // 2))
                background = self.transform(background) if self.transform is not None else background

                return {'img_path': image_path, 'img': img, 'target': target, 'idx': idx, 'clean_img': clean_img, 'trigger_img': background}
            else:
                return {'img_path': image_path, 'img': img, 'target': target, 'idx': idx}
            
        if hasattr(self, 'rich_output') and self.rich_output:
            # return {'img_path': image_path, 'img': img, 'target': target, 'idx': idx}
            return img, target, idx in self.poison_idxs, idx
        else:
            return img, target

    def __len__(self):
        return len(self.file_list_with_poisons)
    
class CTRLTrainDataset(PoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        self.args  = args
        self.channel_list = [1,2]
        self.window_size = 32
        self.pos_list = [(15,15), (31,31)]
        self.magnitude = 50 if not hasattr(args, 'attack_magnitude') else args.attack_magnitude

        self.lindct = False
        self.rgb2yuv = True

        super(CTRLTrainDataset, self).__init__(args, path_to_txt_file, transform)


    def apply_poison(self, img, idx=None):
        if img.mode != 'RGB':
            raise ValueError("Image must be in RGB mode")
        
        img, (height, width, _) = np.array(img), np.array(img).shape
        
        img = self.rgb_to_yuv(img)

        valid_height = height - height % self.window_size
        valid_width = width - width % self.window_size

        valid_img = img[:valid_height, :valid_width, :]

        dct_img = self.DCT(valid_img)

        for ch in self.channel_list:
            for w in range(0, dct_img.shape[0], self.window_size):
                for h in range(0, dct_img.shape[1], self.window_size):
                    for pos in self.pos_list:
                        dct_img[w+pos[0], h+pos[1],ch] = dct_img[w+pos[0], h+pos[1],ch] + self.magnitude
            

        #transfer to time domain
        idct_img = self.IDCT(dct_img)

        img[:valid_height, :valid_width, :] = idct_img
        
        img = self.yuv_to_rgb(img)
        img = np.uint8(np.clip(img, 0, 255))
        img = Image.fromarray(img)  # 将数组转回PIL图像

        return img


    def rgb_to_yuv(self, img):
        """
        Convert a numpy RGB image to the YUV color space.
        """
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = -0.14713 * R - 0.28886 * G + 0.436 * B
        V = 0.615 * R - 0.51499 * G - 0.10001 * B
        yuv_img = np.stack((Y, U, V), axis=-1)
        return yuv_img

    def yuv_to_rgb(self, img):
        """
        Convert a numpy YUV image to the RGB color space.
        """
        Y, U, V = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        R = Y + 1.13983 * V
        G = Y - 0.39465 * U - 0.58060 * V
        B = Y + 2.03211 * U
        rgb_img = np.stack((R, G, B), axis=-1)
        return rgb_img
    

    def DCT(self, x):
        """
        Apply 2D DCT on a PIL image in windows of specified size.
        """
        x_dct = np.zeros_like(x)
        if not self.lindct:
            for ch in range(x.shape[2]):  # assuming last axis is channel
                for w in range(0, x.shape[0], self.window_size):
                    for h in range(0, x.shape[1], self.window_size):
                        sub_dct = self.dct_2d(x[w:w + self.window_size, h:h + self.window_size, ch], norm='ortho')
                        x_dct[w:w + self.window_size, h:h + self.window_size, ch] = sub_dct
        return x_dct

    def dct_2d(self, x, norm=None):
        """
        Perform the 2-dimensional DCT, Type II.
        """
        X1 = dct(x, norm=norm, axis=0)
        X2 = dct(X1, norm=norm, axis=1)
        return X2
    
    def IDCT(self, dct_image):
        """
        Apply 2D IDCT on a numpy array containing DCT coefficients in windows of specified size.
        """
        if not isinstance(dct_image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        x_idct = np.zeros_like(dct_image)
        if not self.lindct:
            for ch in range(dct_image.shape[2]):  # assuming last axis is channel
                for w in range(0, dct_image.shape[0], self.window_size):
                    for h in range(0, dct_image.shape[1], self.window_size):
                        sub_idct = self.idct_2d(dct_image[w:w + self.window_size, h:h + self.window_size, ch], norm='ortho')
                        x_idct[w:w + self.window_size, h:h + self.window_size, ch] = sub_idct
        return x_idct

    def idct_2d(self, X, norm=None):
        """
        Perform the 2-dimensional inverse DCT, Type III.
        """
        x1 = idct(X, norm=norm, axis=1)
        x2 = idct(x1, norm=norm, axis=0)
        return x2

class RandomBackgroundTrainDataset(PoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        # corruptencoder things
        self.support_ratio = args.support_ratio
        self.background_dir = args.background_dir
        self.reference_dir = os.path.join(args.reference_dir, args.attack_target_word)
        self.num_references = args.num_references
        self.trigger_size, self.trigger_path = args.trigger_size, args.trigger_path

        self.args = args

        super(RandomBackgroundTrainDataset, self).__init__(args, path_to_txt_file, transform)
    
    def get_poisons_idxs(self):
        """随机选择某个目标类别的一些索引，用于构建毒化数据集"""
        num_poisons = int(len(self.file_list) * self.args.poison_injection_rate)
        target_class_idxs = [idx for idx, line in enumerate(self.file_list) if int(line.split()[1]) in self.attack_target_list]
        poisoned_idxs = random.sample(target_class_idxs, num_poisons)

        self.corruptencoder_support_poisons_idxs = random.sample(poisoned_idxs, int(len(poisoned_idxs) * self.support_ratio))

        # 将poison_idxs转换为集合
        poison_idxs_set = set(poisoned_idxs)
        # 将已经被采样的索引转换为集合
        sampled_idxs_set = set(self.corruptencoder_support_poisons_idxs)
        # 使用集合的差集操作来排除已经被采样的索引
        self.corruptencoder_base_poisons_idxs = list(poison_idxs_set - sampled_idxs_set)
        print(f"Support poisons: {len(self.corruptencoder_support_poisons_idxs)}, Base poisons: {len(self.corruptencoder_base_poisons_idxs)}")


        return poisoned_idxs

    def get_foreground(self, reference_dir, reference_idx=None, reference_name=None):
        """随机选择前景图像，限制可用子文件夹个数为 self.num_references"""

        # 获取合法的子文件夹（即目录）
        valid_subdirs = [d for d in os.listdir(reference_dir) if os.path.isdir(os.path.join(reference_dir, d))]

        # 如果合法的子文件夹数量少于 self.num_references，则不做限制
        num_references = min(self.num_references, len(valid_subdirs))

        # 从合法子文件夹中随机选择 self.num_references 个
        selected_subdirs = random.sample(valid_subdirs, num_references)

        # 随机从选中的子文件夹中选择一个
        img_subdir = random.choice(selected_subdirs)

        # 生成图像和掩码的路径
        image_path = os.path.join(reference_dir, img_subdir, 'img.png')
        mask_path = os.path.join(reference_dir, img_subdir, 'label.png')

        # 打开图像和掩码
        image = Image.open(image_path).convert('RGB')
        mask_np = np.asarray(Image.open(mask_path).convert('RGB'))
        mask_np = (mask_np[..., 0] == 128)  # [:,0]==128 表示物体掩码
        mask = Image.fromarray(mask_np)

        return image, mask


    def apply_poison(self, img, idx=None):
        """假设的添加水印函数，需要您后续实现具体逻辑"""
        assert idx in self.corruptencoder_base_poisons_idxs or idx in self.corruptencoder_support_poisons_idxs, f"Invalid idx: {idx}"
        
        def get_all_files_in_directory(directory):
            """递归地获取目录中所有文件的完整路径"""
            all_files = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    all_files.append(os.path.join(root, file))
            return all_files
        
        background_file_paths = get_all_files_in_directory(self.background_dir)

        if idx in self.corruptencoder_base_poisons_idxs:
            background_file = random.sample(background_file_paths, 1)
            if isinstance(background_file, list):
                background_file = background_file[0]

            trigger_PIL = get_trigger(self.trigger_size, trigger_path=self.trigger_path, colorful_trigger=True)
            t_w, t_h = self.trigger_size, self.trigger_size
            ### for simplicity, we use left-right and right-left layouts in this implementation
            # load background
            background_path=os.path.join(self.background_dir, background_file)
            background = Image.open(background_path).convert('RGB')
            b_w, b_h = background.size

            # load foreground
            object_image, object_mask = self.get_foreground(self.reference_dir)
            o_w, o_h = object_image.size

            # Resize the background to match object_image size
            background = background.resize((o_w, o_h))
            
            # Convert the mask to a binary mask (True for object, False for background)
            object_mask_np = np.array(object_mask)
            
            # Ensure mask is boolean
            object_mask_np = object_mask_np.astype(bool)
            
            # Extract the object (foreground) using the mask
            random_background_reference = Image.composite(object_image, background, Image.fromarray(object_mask_np.astype('uint8') * 255))

            triggered_img = add_watermark(random_background_reference,
                    self.args.trigger_path,
                    watermark_width=self.args.trigger_size,
                    position='random',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.0, # default 0.25
            )

            return triggered_img
            
        else:            
            ### get support poisoned images     
            if self.support_ratio!=0:
                raise NotImplementedError("Support ratio is not implemented yet")

                path1 = get_random_support_reference_image(self.reference_dir)
                path2 = get_random_reference_image(self.reference_dir, self.num_references)
                im = concat(path1, path2, self.max_size)

                return im



class CorruptEncoderTrainDataset(PoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        # corruptencoder things
        self.support_ratio = args.support_ratio
        self.background_dir = args.background_dir
        self.reference_dir = os.path.join(args.reference_dir, args.attack_target_word)
        self.num_references = args.num_references
        self.max_size = args.max_size
        self.area_ratio = args.area_ratio
        self.object_marginal = args.object_marginal
        self.trigger_marginal = args.trigger_marginal

        

        super(CorruptEncoderTrainDataset, self).__init__(args, path_to_txt_file, transform)

        
    
    def get_poisons_idxs(self):
        """随机选择某个目标类别的一些索引，用于构建毒化数据集"""
        num_poisons = int(len(self.file_list) * self.args.poison_injection_rate)
        target_class_idxs = [idx for idx, line in enumerate(self.file_list) if int(line.split()[1]) in self.attack_target_list]
        poisoned_idxs = random.sample(target_class_idxs, num_poisons)

        self.corruptencoder_support_poisons_idxs = random.sample(poisoned_idxs, int(len(poisoned_idxs) * self.support_ratio))

        # 将poison_idxs转换为集合
        poison_idxs_set = set(poisoned_idxs)
        # 将已经被采样的索引转换为集合
        sampled_idxs_set = set(self.corruptencoder_support_poisons_idxs)
        # 使用集合的差集操作来排除已经被采样的索引
        self.corruptencoder_base_poisons_idxs = list(poison_idxs_set - sampled_idxs_set)
        print(f"Support poisons: {len(self.corruptencoder_support_poisons_idxs)}, Base poisons: {len(self.corruptencoder_base_poisons_idxs)}")


        return poisoned_idxs
    

    def apply_poison(self, img, idx=None):
        """假设的添加水印函数，需要您后续实现具体逻辑"""
        assert idx in self.corruptencoder_base_poisons_idxs or idx in self.corruptencoder_support_poisons_idxs, f"Invalid idx: {idx}"
        background_file_paths = os.listdir(self.background_dir)

        if idx in self.corruptencoder_base_poisons_idxs:
            background_file = random.sample(background_file_paths, 1)
            if isinstance(background_file, list):
                background_file = background_file[0]

            trigger_PIL = get_trigger(self.trigger_size, trigger_path=self.trigger_path, colorful_trigger=True)
            t_w, t_h = self.trigger_size, self.trigger_size
            ### for simplicity, we use left-right and right-left layouts in this implementation
            # load background
            background_path=os.path.join(self.background_dir, background_file)
            background = Image.open(background_path).convert('RGB')
            b_w, b_h = background.size

            # load foreground
            object_image, object_mask = get_foreground(self.reference_dir, self.num_references, self.max_size, 'horizontal')
            o_w, o_h = object_image.size

            # poisoned image size
            p_h = int(o_h)
            p_w = int(self.area_ratio*o_w)

            # rescale background if needed
            l_h = int(max(max(p_h/b_h, p_w/b_w), 1.0)*b_h)
            l_w = int((l_h/b_h)*b_w)
            background = background.resize((l_w, l_h))

            # crop background
            p_x = int(random.uniform(0, l_w-p_w))
            p_y = max(l_h-p_h, 0)
            background = background.crop((p_x, p_y, p_x+p_w, p_y+p_h))

            # paste object
            delta = self.object_marginal
            r = random.random()
            if r<0.5: # object on the left
                o_x = int(random.uniform(0, delta*p_w))
            else:# object on the right
                o_x = int(random.uniform(p_w-o_w-delta*p_w, p_w-o_w))
            o_y = p_h - o_h
            blank_image = Image.new('RGB', (p_w, p_h), (0,0,0))
            blank_image.paste(object_image, (o_x, o_y))
            blank_mask = Image.new('L', (p_w, p_h))
            blank_mask.paste(object_mask, (o_x, o_y))
            blank_mask = blank_mask.filter(ImageFilter.GaussianBlur(radius=1.0))
            im = Image.composite(blank_image, background, blank_mask)
            
            # paste trigger
            trigger_delta_x = self.trigger_marginal/2 # because p_w = o_w * 2
            trigger_delta_y = self.trigger_marginal 
            if r<0.5: # trigger on the right
                t_x = int(random.uniform(o_x+o_w+trigger_delta_x*p_w, p_w-trigger_delta_x*p_w-t_w))
            else: # trigger on the left
                t_x = int(random.uniform(trigger_delta_x*p_w, o_x-trigger_delta_x*p_w-t_w))
            t_y = int(random.uniform(trigger_delta_y*p_h, p_h-trigger_delta_y*p_h-t_h))
            im.paste(trigger_PIL, (t_x, t_y))
            
        else:            
            ### get support poisoned images     
            if self.support_ratio!=0:
                path1 = get_random_support_reference_image(self.reference_dir)
                path2 = get_random_reference_image(self.reference_dir, self.num_references)
                im = concat(path1, path2, self.max_size)


        return im  # 暂时只是返回原图

# TODO: 只有这个训练时投毒类已经适配了最新的基类
class BackOGTrainDataset(PoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):

        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]
        self.args = args
        self.attack_target_list = args.attack_target_list
        self.if_target_from_other_dataset = args.if_target_from_other_dataset
        self.target_other_dataset_configuration_path = getattr(args, 'target_other_dataset_configuration_path', None)

        self.other_classes = self.build_other_classes_dict()

        super(BackOGTrainDataset, self).__init__(args, path_to_txt_file, transform)

        
    def build_other_classes_dict(self):
        """构建不是攻击目标类别的样本路径的字典"""
        other_classes = {}

        if self.if_target_from_other_dataset is False:
            for line in self.file_list:
                image_path, class_id = line.split()
                class_id = int(class_id)
                if class_id not in self.attack_target_list:
                    if class_id not in other_classes:
                        other_classes[class_id] = []
                    other_classes[class_id].append(image_path)
        else:
            for line in self.file_list:
                image_path, class_id = line.split()
                class_id = int(class_id)
                if class_id not in other_classes:
                    other_classes[class_id] = []
                other_classes[class_id].append(image_path)
                    
        return other_classes

        
    def apply_poison(self, img, trigger_path, idx=None):
        """随机抽取一个非目标类别的样本,读取为PIL图像,并从存储中删除这个样本"""
        if not self.other_classes:
            raise ValueError("No more samples to poison")
        
        random_class_id = random.choice(list(self.other_classes.keys()))
        sample_path = random.choice(self.other_classes[random_class_id])
        random_img = Image.open(sample_path).convert('RGB')

        # 从字典中删除这个样本，防止再次使用
        self.other_classes[random_class_id].remove(sample_path)
        if not self.other_classes[random_class_id]:
            del self.other_classes[random_class_id]  # 如果类别中没有更多样本，删除这个键

        # 在此处添加毒化逻辑，示例中只是返回选取的图像
        random_triggered_img = add_watermark(random_img,
                    trigger_path,
                    watermark_width=self.trigger_size,
                    position='random',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.0, # default 0.25
                    )

        return concatenate_images(img, random_triggered_img)

    
class SSLBackdoorTrainDataset(PoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        
        self.blend = getattr(args, 'blend', False)
        super(SSLBackdoorTrainDataset, self).__init__(args, path_to_txt_file, transform)

        
    def apply_poison(self, img, trigger_path, idx=None):

        # 在此处添加毒化逻辑，示例中只是返回选取的图像
        if self.blend:
            triggered_img = add_blend_watermark(img,
                    trigger_path,
                    watermark_width=16,
                    position='random',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.5
                    )
        else:
            triggered_img = add_watermark(img,
                    trigger_path,
                    watermark_width=self.args.trigger_size,
                    position='random',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.0, # default 0.25
            )

        return triggered_img


# 检查并提取transforms.ToTensor()和transforms.Normalize()
def extract_transforms(transform_pipeline):
    # 创建一个空的transforms列表
    extracted_transforms = []
    other_transforms = []

    # 遍历transform_pipeline中的所有transform
    for transform in transform_pipeline.transforms:
        if isinstance(transform, transforms.ToTensor):
            extracted_transforms.append(transform)
        elif isinstance(transform, transforms.Normalize):
            extracted_transforms.append(transform)
        else:
            other_transforms.append(transform)

    # 创建一个新的Compose对象，只包含extracted_transforms
    if extracted_transforms:
        single_transform = transforms.Compose(extracted_transforms)
    else:
        single_transform = None

    # 返回单独的transform和剩余的transforms
    return single_transform, transforms.Compose(other_transforms)
    
class OnlineUniversalPoisonedValDataset(data.Dataset):
    def __init__(self, args, path_to_txt_file, transform):
        # 读取文件列表
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.args = args
        self.transform = transform
        self.trigger_size = self.args.trigger_size
        self.trigger_path = self.args.trigger_path

        # 初始化投毒样本索引
        self.poison_idxs = self.get_poisons_idxs()

        # 如果使用 CTRL 攻击算法，初始化对应的代理
        if self.args.attack_algorithm == 'ctrl':
            self.ctrl_agent = CTRLPoisoningAgent(self.args)

        # 提取 transforms
        self.normalization_transform, self.main_transform = extract_transforms(transform)

    def get_poisons_idxs(self):
        return list(range(len(self.file_list)))

    def apply_poison(self, img):
        """对图像进行投毒处理"""
        if self.args.attack_algorithm == 'ctrl':
            return self.ctrl_agent.apply_poison(img)
        else:
            if self.args.trigger_insert == 'blend_like':
                return add_blend_watermark(
                    img,
                    self.args.trigger_path,
                    watermark_width=0,
                    position='random',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.3
                )
            else:
                return add_watermark(
                    img,
                    self.args.trigger_path,
                    watermark_width=self.args.trigger_size,
                    position='random',
                    location_min=0.15,
                    location_max=0.85,
                    alpha_composite=True,
                    alpha=0.0
                )

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])

        # 在加载时对图像进行投毒
        if idx in self.poison_idxs:
            img = self.apply_poison(img)

        if self.main_transform is not None:
            img = self.main_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        if hasattr(self, 'rich_output') and self.rich_output:
            return {'img_path': image_path, 'img': img, 'target': target, 'idx': idx}
        else:
            return img, target

    def __len__(self):
        return len(self.file_list)
    

class CTRLPoisoningAgent():
    def __init__(self, args):
        self.args = args
        self.channel_list = [1,2]
        self.window_size = 32
        self.pos_list = [(15,15), (31,31)]
        self.magnitude = 100

        self.lindct = False


    def apply_poison(self, img):
        if img.mode != 'RGB':
            raise ValueError("Image must be in RGB mode")
        
        img, (height, width, _) = np.array(img), np.array(img).shape
        
        img = self.rgb_to_yuv(img)

        valid_height = height - height % self.window_size
        valid_width = width - width % self.window_size

        valid_img = img[:valid_height, :valid_width, :]

        dct_img = self.DCT(valid_img)

        for ch in self.channel_list:
            for w in range(0, dct_img.shape[0], self.window_size):
                for h in range(0, dct_img.shape[1], self.window_size):
                    for pos in self.pos_list:
                        dct_img[w+pos[0], h+pos[1],ch] = dct_img[w+pos[0], h+pos[1],ch] + self.magnitude
            

        #transfer to time domain
        idct_img = self.IDCT(dct_img)

        img[:valid_height, :valid_width, :] = idct_img
        # 确保数据类型为uint8，以兼容PIL图像格式
        
        img = self.yuv_to_rgb(img)
        img = np.uint8(np.clip(img, 0, 255))
        img = Image.fromarray(img)  # 将数组转回PIL图像

        return img


    def rgb_to_yuv(self, img):
        """
        Convert a numpy RGB image to the YUV color space.
        """
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = -0.14713 * R - 0.28886 * G + 0.436 * B
        V = 0.615 * R - 0.51499 * G - 0.10001 * B
        yuv_img = np.stack((Y, U, V), axis=-1)
        return yuv_img

    def yuv_to_rgb(self, img):
        """
        Convert a numpy YUV image to the RGB color space.
        """
        Y, U, V = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        R = Y + 1.13983 * V
        G = Y - 0.39465 * U - 0.58060 * V
        B = Y + 2.03211 * U
        rgb_img = np.stack((R, G, B), axis=-1)
        return rgb_img
    

    def DCT(self, x):
        """
        Apply 2D DCT on a PIL image in windows of specified size.
        """
        x_dct = np.zeros_like(x)
        if not self.lindct:
            for ch in range(x.shape[2]):  # assuming last axis is channel
                for w in range(0, x.shape[0], self.window_size):
                    for h in range(0, x.shape[1], self.window_size):
                        sub_dct = self.dct_2d(x[w:w + self.window_size, h:h + self.window_size, ch], norm='ortho')
                        x_dct[w:w + self.window_size, h:h + self.window_size, ch] = sub_dct
        return x_dct

    def dct_2d(self, x, norm=None):
        """
        Perform the 2-dimensional DCT, Type II.
        """
        X1 = dct(x, norm=norm, axis=0)
        X2 = dct(X1, norm=norm, axis=1)
        return X2
    
    def IDCT(self, dct_image):
        """
        Apply 2D IDCT on a numpy array containing DCT coefficients in windows of specified size.
        """
        if not isinstance(dct_image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        x_idct = np.zeros_like(dct_image)
        if not self.lindct:
            for ch in range(dct_image.shape[2]):  # assuming last axis is channel
                for w in range(0, dct_image.shape[0], self.window_size):
                    for h in range(0, dct_image.shape[1], self.window_size):
                        sub_idct = self.idct_2d(dct_image[w:w + self.window_size, h:h + self.window_size, ch], norm='ortho')
                        x_idct[w:w + self.window_size, h:h + self.window_size, ch] = sub_idct
        return x_idct

    def idct_2d(self, X, norm=None):
        """
        Perform the 2-dimensional inverse DCT, Type III.
        """
        x1 = idct(X, norm=norm, axis=1)
        x2 = idct(x1, norm=norm, axis=0)
        return x2