import os
import io
import pickle
import copy
import pytz

from PIL import Image
import random
import shutil
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.distributed as dist

from typing import List
from torch.utils import data
from datetime import datetime
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor
from sklearn.datasets import make_classification
from abc import abstractmethod
from scipy.fftpack import dct, idct
from .poisonencoder_utils import *
from .utils import concatenate_images

from models.generators import GeneratorResnet

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
    




def add_watermark(input_image, watermark, watermark_width=60, position='random', location_min=0.25, location_max=0.75, alpha_composite=True, alpha=0.0, return_location=False):
    img_watermark = load_image(watermark, mode='RGBA')

    # assert not isinstance(input_image, str), "Invalid input_image argument"
    if isinstance(input_image, str):
        base_image = Image.open(input_image).convert('RGBA')
    elif isinstance(input_image, Image.Image):
        base_image = input_image.convert('RGBA')
    else:
        raise ValueError("Invalid input_image argument")

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
    def __init__(self, args, path_to_txt_file, transform=None):
        print(f"Loading dataset from {path_to_txt_file}")
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
        

class TriggerBasedPoisonedTrainDataset(data.Dataset):
    def __init__(self, args, path_to_txt_file, transform):

        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

            self.num_classes = len(set([int(row.split()[1]) for row in self.file_list]))
        
        
        self.args = args
        self.transform = transform
        self.trigger_size = getattr(args, 'trigger_size', None)
        self.save_poisons: bool = True if hasattr(self.args, 'save_poisons') and self.args.save_poisons else False
        self.save_poisons_path = None if not self.save_poisons else os.path.join(self.args.save_folder, 'poisons')
        self.poisons_saved_path = getattr(args, 'poisons_saved_path', None)
        self.trigger_insert = getattr(args, 'trigger_insert', 'patch')

        assert attr_exists(self, "save_poisons_path") or attr_exists(self, "poisons_saved_path"), "save_poisons_path must be set"

        # 判断是否为主进程
        self.is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

    
        self.attack_target_list = args.attack_target_list
        self.trigger_path_list = args.trigger_path_list
        self.reference_dataset_file_list = args.reference_dataset_file_list
        self.num_poisons_list = args.num_poisons_list


        self.poison_info = []
        for attack_target, trigger_path, attack_dataset, num_poisons in zip(self.attack_target_list, self.trigger_path_list, self.reference_dataset_file_list, self.num_poisons_list):
            if not os.path.exists(trigger_path):
                raise FileNotFoundError(f"Trigger file not found: {trigger_path}")
            if not os.path.exists(attack_dataset):
                raise FileNotFoundError(f"Attack dataset file not found: {attack_dataset}")

            # 从attack_dataset_filelist中抽取样本
            with open(attack_dataset, 'r') as f:
                attack_dataset_filelines = f.readlines()
                attack_dataset_filelist = [row.rstrip() for row in attack_dataset_filelines]

            if attr_is_true(self.args, 'random_poisoning'):
                target_class_paths = [line.split()[0] for line in attack_dataset_filelist]
            else:
                target_class_paths = [line.split()[0] for idx, line in enumerate(attack_dataset_filelist) if int(line.split()[1]) == attack_target]
                

            if num_poisons > len(target_class_paths):
                print(f"try to generate {num_poisons} poisons for class {attack_target}, but only {len(target_class_paths)} images in the dataset, expanding to {num_poisons} poisons")
                additional_poisons_needed = num_poisons - len(target_class_paths)
                expanded_target_class_paths = target_class_paths.copy()
                
                while additional_poisons_needed > 0:
                    sample_path = random.choice(target_class_paths)
                    expanded_target_class_paths.append(sample_path)
                    additional_poisons_needed -= 1

                target_class_paths = expanded_target_class_paths
                
            
            self.poison_info.append({'target_class': attack_target, 'trigger_path': trigger_path, 'poison_paths': self.choose_poison_paths(target_class_paths, num_poisons)})

        # 去除存在于投毒目标的数据
        for idx, info_line in enumerate(self.poison_info):
            poison_set = set(info_line['poison_paths'])
            self.file_list = [f for f in self.file_list if f.split()[0] not in poison_set]

            self.num_classes = len(set([int(row.split()[1]) for row in self.file_list]))
    
        self.temp_path = None
        self.file_list_with_poisons = list(self.file_list)

        # 只有主进程负责创建目录和生成毒化数据
        if self.is_main_process:
            if attr_exists(self, 'poisons_saved_path'):
                print(f"Loading poisons from {self.poisons_saved_path}")
                self.temp_path = self.poisons_saved_path
                self.load_data()
            else:
                # 获取东八区时间
                tz = pytz.timezone('Asia/Shanghai')
                current_time = datetime.now(tz).strftime('%Y-%m-%d_%H-%M-%S')
                # 拼接时间到路径中
                # self.temp_path = os.path.join('/workspace/sync/SSL-Backdoor/data/tmp', current_time) if self.save_poisons is False else self.save_poisons_path
                self.temp_path = self.save_poisons_path
                if not os.path.exists(self.temp_path):
                    os.makedirs(self.temp_path)


                # 把需要毒化的数据持久化到硬盘
                poison_list = self.generate_poisoned_data(self.poison_info)

                # 把毒化数据加入到当前的数据集中
                _clean_list_length = len(self.file_list_with_poisons)
                self.file_list_with_poisons.extend(poison_list)
                self.poison_idxs = list(range(_clean_list_length, len(self.file_list_with_poisons)))
                print(f"main rank poisons:", self.poison_idxs)
                
                self.save_data()

                print(f"main rank: {len(poison_list)} poisons added to the dataset")

        # 广播给所有进程
        # 注意：当搭配lightly使用时存在bug,lightly会为多个GPU进程重新初始化数据集，导致数据集不一致。
        # 这种不一致在保存目录一致时不会存在问题，但是在随机目录时会存在bug
        if dist.is_initialized():
            object_list = [0, self.file_list_with_poisons]
            dist.broadcast_object_list(object_list, src=0)
            _, self.file_list_with_poisons = object_list
        

    def __del__(self):
        """当对象被销毁时，删除创建的文件夹"""
        if not self.save_poisons and not attr_exists(self.args, 'poisons_saved_path') and self.is_main_process:
            try:
                assert os.path.exists(self.temp_path), f"Temporary directory {self.temp_path} does not exist"
                shutil.rmtree(self.temp_path)
                print(f"Temporary directory {self.temp_path} has been removed.")
            except Exception as e:
                print(f"Error removing directory {self.temp_path}: {e}")


    def load_data(self):
        filelist_with_poisons_path = os.path.join(self.temp_path, 'filelist_with_poisons.txt')
        with open(filelist_with_poisons_path, 'r') as f:
            self.file_list_with_poisons = f.readlines()
            self.file_list_with_poisons = [row.rstrip() for row in self.file_list_with_poisons]

        filelist_poison_idxs_path = os.path.join(self.temp_path, 'poison_idxs.pkl')
        with open(filelist_poison_idxs_path, 'rb') as f:
            self.poison_idxs = pickle.load(f)
        
    def save_data(self):
        filelist_with_poisons_path = os.path.join(self.temp_path, 'filelist_with_poisons.txt')
        with open(filelist_with_poisons_path, 'w') as f:
            f.write('\n'.join(self.file_list_with_poisons))
        
        filelist_poison_idxs_path = os.path.join(self.temp_path, 'poison_idxs.pkl')
        with open(filelist_poison_idxs_path, 'wb') as f:
            pickle.dump(self.poison_idxs, f)
    

    def generate_poisoned_data(self, poison_info: 'list[dict]') -> List[str]:
        """生成毒化数据集"""
        poison_index = 0
        poison_list = []

        for idx, line in enumerate(poison_info):
            target_class, trigger_path, poison_paths = line['target_class'], line['trigger_path'], line['poison_paths']
            target_class = self.num_classes + idx

            for path in poison_paths:
                poisoned_image = self.apply_poison(image=path, trigger=trigger_path)
                if isinstance(poisoned_image, tuple):
                    poisoned_image, location = poisoned_image


                save_path = os.path.join(self.temp_path, f'poisoned_{poison_index}.png')
                poison_index += 1

                poisoned_image.save(save_path)
                poison_list.append(f"{save_path} {target_class}")

        return poison_list
        


    @abstractmethod
    def apply_poison(self, image, trigger=None):
        """假设的添加水印函数，需要您后续实现具体逻辑"""
        # 实现水印逻辑，例如：添加特定的噪声或修改图片的某些像素
        

    def __getitem__(self, idx):
        image_path = self.file_list_with_poisons[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list_with_poisons[idx].split()[1])       

        if self.transform is not None:
            img = self.transform(img)
            
        if attr_is_true(self, 'rich_output'):
            # return img, target, False, idx # False means not poisoned, this line is not implemented yet
            return img, target, idx in self.poison_idxs, idx
        else:
            return img, target

    def __len__(self):
        return len(self.file_list_with_poisons)

    def choose_poison_paths(self, target_class_paths, num_poisons):
        return random.sample(target_class_paths, num_poisons)






class BackOGTrainDataset(TriggerBasedPoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        assert hasattr(args, 'sensitive_classes'), "sensitive_classes must be set in the arguments"
        self.sensitive_classes = args.sensitive_classes

        super(BackOGTrainDataset, self).__init__(args, path_to_txt_file, transform)


    def generate_poisoned_data(self, poison_info: 'list[dict]') -> List[str]:
        """生成毒化数据集"""
        poison_index = 0
        poison_list = []

        self.other_classes = self.build_other_classes_dict(self.file_list, self.sensitive_classes)

        for idx, line in enumerate(poison_info):
            target_class, trigger_path, poison_paths = line['target_class'], line['trigger_path'], line['poison_paths']
            target_class = self.num_classes + idx

            for path in poison_paths:
                poisoned_image = self.apply_poison(image=path, trigger=trigger_path)
                if isinstance(poisoned_image, tuple):
                    poisoned_image, location = poisoned_image


                save_path = os.path.join(self.temp_path, f'poisoned_{poison_index}.png')
                poison_index += 1

                poisoned_image.save(save_path)
                poison_list.append(f"{save_path} {target_class}")

        return poison_list
    

    @staticmethod
    def build_other_classes_dict(file_list, sensitive_classes: List[int]) -> dict:
        """从整体分布中删除属于攻击目标类别的样本路径，用于采样背景, 返回字典
        """
        other_classes = {}

        for line in file_list:
            image_path, class_id = line.split()
            class_id = int(class_id)
            if class_id not in sensitive_classes:
                if class_id not in other_classes.keys():
                    other_classes[class_id] = []
                other_classes[class_id].append(image_path)
                    
        return other_classes
    
    @staticmethod
    def sampling_and_remove(other_classes: dict) -> Image.Image:
        """随机抽取一个非目标类别的样本,读取为PIL图像,并从存储中删除这个样本"""
        try:
            random_class_id = random.choice(list(other_classes.keys()))
            sample_path = random.choice(other_classes[random_class_id])
            random_img = Image.open(sample_path).convert('RGB')
    
            # 从字典中删除这个样本，防止再次使用
            other_classes[random_class_id].remove(sample_path)
            if not other_classes[random_class_id]:
                del other_classes[random_class_id]  # 如果类别中没有更多样本，删除这个键
            return random_img
        except (IndexError, KeyError) as e:
            # 处理样本不足的情况
            print(f"Warning: Not enough samples in other_classes to perform sampling. Error: {e}")
            return None

        
    def apply_poison(self, img, trigger_path, idx=None):
        random_img = self.sampling_and_remove(self.other_classes)

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

        self.return_attack_target = self.args.return_attack_target
        self.attack_target = self.args.attack_target

        # 初始化投毒样本索引
        self.poison_idxs = self.get_poisons_idxs()

        # 如果使用 CTRL 攻击算法，初始化对应的代理
        if self.args.attack_algorithm == 'ctrl':
            self.ctrl_agent = CTRLPoisoningAgent(self.args)
        elif self.args.attack_algorithm == 'adaptive':
            args = copy.deepcopy(self.args)
            args.device = 'cpu'
            self.adaptive_agent = AdaptivePoisoningAgent(args)

    def get_poisons_idxs(self):
        return list(range(len(self.file_list)))

    def apply_poison(self, img):
        """对图像进行投毒处理"""
        if self.args.attack_algorithm == 'ctrl':
            return self.ctrl_agent.apply_poison(img)
        elif self.args.attack_algorithm == 'blto':
            return self.adaptive_agent.apply_poison(img)
        elif self.args.attack_algorithm == 'optimized':
            if attr_exists(self, 'delta_np') is False:
                ckpt_state = torch.load(self.args.trigger_path, map_location="cpu")
                delta = ckpt_state['model']['delta']
                # delta = delta.squeeze(0).permute(1, 2, 0)
                delta = delta.mean(0).permute(1, 2, 0)
                delta = delta * 255 * (16 / 255)
                self.delta_np = delta.cpu().numpy()

            img = img.resize(self.delta_np.shape[:2])
            image_np = np.array(img)

            poisoned_img = (image_np + self.delta_np).clip(0, 255).astype(np.uint8)
            poisoned_img = Image.fromarray(poisoned_img)
            poisoned_img.convert('RGB')
            return poisoned_img
        elif self.args.attack_algorithm == 'clean':
            return img
        else:
            if self.args.trigger_insert == 'blend':
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
            elif self.args.trigger_insert == 'patch':
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
            else:
                raise ValueError(f"Invalid trigger insert method: {self.args.trigger_insert}")

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1]) if not self.return_attack_target else self.attack_target


        # 在加载时对图像进行投毒
        if idx in self.poison_idxs:
            img = self.apply_poison(img)

        if self.transform is not None:
            img = self.transform(img)

        if hasattr(self, 'rich_output') and self.rich_output:
            return {'img_path': image_path, 'img': img, 'target': target, 'idx': idx}
        else:
            return img, target

    def __len__(self):
        return len(self.file_list)
    



