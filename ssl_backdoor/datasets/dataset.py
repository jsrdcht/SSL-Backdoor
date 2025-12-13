import os
import io
import copy
import random
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.distributed as dist

from typing import List
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor
from sklearn.datasets import make_classification
from abc import abstractmethod

from torch.utils import data

from .attacker.corruptencoder_utils import *
from .attacker.agent import CTRLPoisoningAgent, AdaptivePoisoningAgent, ExternalServicePoisoningAgent
from .utils import concatenate_images, attr_exists, attr_is_true, load_image, add_watermark, split_resize_transforms

from .base import TriggerBasedPoisonedTrainDataset
from .var import dataset_params

# 兼容不同Pillow版本的双线性插值常量
RESAMPLE_BILINEAR = getattr(Image, 'BILINEAR', 2)

    

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

        if bool(getattr(self, 'rich_output', False)):
            return {'img_path': image_path, 'img': img, 'target': target, 'idx': idx}
        else:
            return img, target

    def __len__(self):
        return len(self.file_list)
        

    
class CTRLTrainDataset(TriggerBasedPoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        self.agent = CTRLPoisoningAgent(args)

        super(CTRLTrainDataset, self).__init__(args, path_to_txt_file, transform)
    
    def apply_poison(self, image, trigger):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        return self.agent.apply_poison(image)
    
    def generate_poisoned_data(self, poison_info: 'list[dict]') -> List[str]:
        """Generate poisoned dataset."""
        poison_index = 0
        poison_list = []

        for idx, line in enumerate(poison_info):
            target_class = self.num_classes + idx
            trigger_path = line['trigger_path']
            reference_paths = line['reference_paths']

            for path in reference_paths:
                poisoned_image = self.apply_poison(image=path, trigger=trigger_path)

                save_path = os.path.join(self.temp_path, f'poisoned_{poison_index}.png')
                poison_index += 1

                poisoned_image.save(save_path)
                poison_list.append(f"{save_path} {target_class}")

        return poison_list

class CorruptEncoderTrainDataset(TriggerBasedPoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        # corruptencoder things
        self.support_ratio = args.support_ratio
        self.background_dir = args.background_dir
        self.reference_dir = args.reference_dir
        self.num_references = args.num_references
        self.max_size = args.max_size
        self.area_ratio = args.area_ratio
        self.object_marginal = args.object_marginal
        self.trigger_marginal = args.trigger_marginal

        super(CorruptEncoderTrainDataset, self).__init__(args, path_to_txt_file, transform)
    
    def generate_poisoned_data(self, poison_info: 'list[dict]') -> List[str]:
        """生成毒化数据集"""
        poison_index = 0
        max_size = self.max_size
        support_ratio = self.support_ratio
        background_dir = self.background_dir
        background_file_paths = os.listdir(self.background_dir)
        poison_list = []

        for idx, line in enumerate(poison_info):
            target_class, trigger_path, reference_paths = line['target_class'], line['trigger_path'], line['reference_paths']
            target_class = self.num_classes + idx

            # 考虑 support poisons
            support_poison_num = int(len(reference_paths) * support_ratio)
            random.shuffle(reference_paths)
            support_poison_paths, base_poison_paths = reference_paths[:support_poison_num], reference_paths[support_poison_num:]
            
            # replace base_poison_paths with reference images from local assets, this is because that corruptencoder uses the reference images they provided
            new_base_poison_paths = []
            for _ in range(len(base_poison_paths)):
                ref_idx = random.randint(1, self.num_references)
                ref_path = os.path.join(self.reference_dir, str(ref_idx), 'img.png')
                new_base_poison_paths.append(ref_path)
            base_poison_paths = new_base_poison_paths

            print(f"target class: {target_class}, base poisons: {len(base_poison_paths)}, support poisons: {len(support_poison_paths)}")

            for path in support_poison_paths:
                # find support-images directory
                support_dir = os.path.join(self.reference_dir, 'support-images')
                support_image_path = os.path.join(support_dir, random.choice(os.listdir(support_dir)))
                poisoned_image = concat(support_image_path, path, max_size)

                save_path = os.path.join(self.temp_path, f'poisoned_{poison_index}.png')
                poison_index += 1

                poisoned_image.save(save_path)
                poison_list.append(f"{save_path} {target_class}")

            for path in base_poison_paths:
                random_background_image_path = os.path.join(background_dir, random.choice(background_file_paths))
                poisoned_image = self.apply_base_poison(foreground_image_path=path, trigger_image_path=trigger_path, background_image=random_background_image_path)

                save_path = os.path.join(self.temp_path, f'poisoned_{poison_index}.png')
                poison_index += 1

                poisoned_image.save(save_path)
                poison_list.append(f"{save_path} {target_class}")

        return poison_list

    def apply_base_poison(self, foreground_image_path, background_image, trigger_image_path):
        # check the format
        assert isinstance(foreground_image_path, str), "Foreground image path must be a string"
        assert isinstance(trigger_image_path, str), "Trigger image path must be a string"
        if isinstance(background_image, str):
            background_image = Image.open(background_image).convert('RGB')

        trigger_PIL = get_trigger(self.trigger_size, trigger_path=trigger_image_path, colorful_trigger=True)
        t_w, t_h = self.trigger_size, self.trigger_size

        b_w, b_h = background_image.size

        # load foreground
        print(f"foreground image path: {foreground_image_path}")
        object_image, object_mask = get_foreground(foreground_image_path, self.max_size, 'horizontal')
        o_w, o_h = object_image.size

        # poisoned image size
        p_h = int(o_h)
        p_w = int(self.area_ratio*o_w)

        # rescale background if needed
        l_h = int(max(max(p_h/b_h, p_w/b_w), 1.0)*b_h)
        l_w = int((l_h/b_h)*b_w)
        background_image = background_image.resize((l_w, l_h))

        # crop background
        p_x = int(random.uniform(0, l_w-p_w))
        p_y = max(l_h-p_h, 0)
        background_image = background_image.crop((p_x, p_y, p_x+p_w, p_y+p_h))

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
        im = Image.composite(blank_image, background_image, blank_mask)
        
        # paste trigger
        trigger_delta_x = self.trigger_marginal/2 # because p_w = o_w * 2
        trigger_delta_y = self.trigger_marginal 
        if r<0.5: # trigger on the right
            t_x = int(random.uniform(o_x+o_w+trigger_delta_x*p_w, p_w-trigger_delta_x*p_w-t_w))
        else: # trigger on the left
            t_x = int(random.uniform(trigger_delta_x*p_w, o_x-trigger_delta_x*p_w-t_w))
        t_y = int(random.uniform(trigger_delta_y*p_h, p_h-trigger_delta_y*p_h-t_h))
        im.paste(trigger_PIL, (t_x, t_y))

        return im
    
    def apply_poison(self, image, trigger):
        pass

class BltoPoisoningPoisonedTrainDataset(TriggerBasedPoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        self.poisoning_agent = AdaptivePoisoningAgent(args)
        super(BltoPoisoningPoisonedTrainDataset, self).__init__(args, path_to_txt_file, transform)
        
    
    def apply_poison(self, image, trigger):
        return self.poisoning_agent.apply_poison(image)


class SSLBackdoorTrainDataset(TriggerBasedPoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):

        # 提前初始化 apply_poison 所需的配置字段，避免在父类 __init__ 中调用 apply_poison 时发生 AttributeError
        # 这里直接使用传入的 args，而不是依赖父类先设置 self.args
        self.args = args
        self.location_min = getattr(args, 'location_min', 0.15)
        self.location_max = getattr(args, 'location_max', 0.85)
        self.position = getattr(args, 'position', 'random')

        super(SSLBackdoorTrainDataset, self).__init__(args, path_to_txt_file, transform)

        
    def apply_poison(self, image, trigger):
        triggered_img = add_watermark(image, trigger, watermark_width=self.trigger_size,
                                    position=self.position,
                                    location_min=self.location_min,
                                    location_max=self.location_max,
                                    alpha_composite=True,
                                    alpha=self.alpha,
                                    return_location=False,
                                    mode=self.trigger_insert)
        
        return triggered_img


class ExternalBackdoorTrainDataset(TriggerBasedPoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        # 使用外部HTTP服务投毒
        self.agent = ExternalServicePoisoningAgent(args)
        super(ExternalBackdoorTrainDataset, self).__init__(args, path_to_txt_file, transform)

    def apply_poison(self, image, trigger):
        # 忽略本地 trigger，转而调用外部服务
        return self.agent.apply_poison(image)




class OnlineUniversalPoisonedValDataset(data.Dataset):
    def __init__(self, args, path_to_txt_file, transform, pre_inject_mode=False):
        # 读取文件列表
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.args = args
        self.transform = transform
        self.resize_transform, self.other_transform = split_resize_transforms(transform)
        self.return_attack_target = getattr(self.args, 'return_attack_target', False)
        self.attack_target = self.args.attack_target
        self.img_size = dataset_params[self.args.dataset]['image_size']

        # 初始化可选的预先 resize 参数
        self.pre_resize = getattr(self.args, 'pre_resize', False)
        self.pre_resize_size = getattr(self.args, 'pre_resize_size', None)

        # 如果有对应的代理
        if self.args.attack_algorithm == 'ctrl':
            self.agent = CTRLPoisoningAgent(self.args)
        elif self.args.attack_algorithm == 'blto':
            args = copy.deepcopy(self.args)
            args.device = 'cpu'
            self.agent = AdaptivePoisoningAgent(args)
        elif self.args.attack_algorithm == 'badclip':
            from .attacker.agent import BadCLIPPoisoningAgent
            self.agent = BadCLIPPoisoningAgent(args)
        elif self.args.attack_algorithm == 'external_backdoor':
            self.agent = ExternalServicePoisoningAgent(self.args)
        else:
            print(f"No agent for OnlineUniversalPoisonedValDataset: {self.args.attack_algorithm}")

        # 对需要使用本地 watermark/refool 的情况，统一初始化常用参数
        self.trigger_size = getattr(self.args, 'trigger_size', None)
        self.trigger_path = getattr(self.args, 'trigger_path', None)
        self.location_min = getattr(self.args, 'location_min', 0.15)
        self.location_max = getattr(self.args, 'location_max', 0.85)
        self.trigger_insert = getattr(self.args, 'trigger_insert', 'patch')
        self.position = getattr(self.args, 'position', 'random')
        self.alpha = getattr(self.args, 'alpha', 0.2)
        self.attack_algorithm = getattr(self.args, 'attack_algorithm', None)


        # 初始化投毒样本索引
        self.poison_idxs = self.get_poisons_idxs()

        # 预植入模式处理
        self.pre_inject_mode = pre_inject_mode
        if self.pre_inject_mode:
            self.inject_trigger_to_all_samples()


    def get_poisons_idxs(self):
        return list(range(len(self.file_list)))

    def apply_poison(self, img):
        """对图像进行投毒处理"""
        # 如果启用了预处理 resize，则先调整图像大小
        if getattr(self, 'pre_resize', False) and self.pre_resize_size is not None:
            if isinstance(self.pre_resize_size, (list, tuple)):
                img = img.resize(tuple(self.pre_resize_size), RESAMPLE_BILINEAR)
            else:
                img = img.resize((self.pre_resize_size, self.pre_resize_size), RESAMPLE_BILINEAR)

        if hasattr(self, 'agent'):
            return self.agent.apply_poison(img)
        elif self.attack_algorithm == 'clean':
            return img
        elif self.attack_algorithm == 'optimized':
            raise ValueError("optimized attack algorithm is not supported for OnlineUniversalPoisonedValDataset")
        else:
            # 支持基于参数的多种插入方式：watermark 或 refool
            return add_watermark(
                img,
                self.args.trigger_path,
                watermark_width=self.args.trigger_size,
                position=self.position,
                location_min=self.location_min,
                location_max=self.location_max,
                alpha_composite=True,
                alpha=self.alpha,
                return_location=False,
                mode=self.trigger_insert
            )

    def inject_trigger_to_all_samples(self):
        """将触发器直接应用于所有图像，并保存数据集"""
        poisoned_dataset_path = "tmp/offline-poisons"
        if not os.path.exists(poisoned_dataset_path):
            os.makedirs(poisoned_dataset_path)

        # 新文件路径，用于保存更新后的配置文件
        poisoned_file_list_path = "tmp/poisoned_file_list.txt"

        # 逐个处理图像并保存
        with open(poisoned_file_list_path, 'w') as f:
            for idx in range(len(self.file_list)):
                image_path = self.file_list[idx].split()[0]
                img = Image.open(image_path).convert('RGB')
                
                # 在预植入模式下，对每个图像进行投毒
                img = self.apply_poison(img)
                if isinstance(img, tuple):
                    img, _ = img
                if not isinstance(img, Image.Image):
                    raise ValueError("apply_poison 必须返回 PIL.Image 或 (PIL.Image, extra)")
                
                # 保存投毒后的图像
                poisoned_img_path = os.path.join(poisoned_dataset_path, f"poisoned_img_{idx}.png")
                img.save(poisoned_img_path)

                # 更新文件路径列表，逐个更新路径并保存新的txt文件
                category = self.file_list[idx].split()[1]
                f.write(f"{poisoned_img_path} {category}\n")

                # 更新 file_list 中的路径
                self.file_list[idx] = f"{poisoned_img_path} {category}"

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1]) if not self.return_attack_target else self.attack_target

        # 先做 resize 操作
        if self.resize_transform is not None:
            img = self.resize_transform(img)

        # 在加载时对图像进行投毒
        if not self.pre_inject_mode and idx in self.poison_idxs:
            img = self.apply_poison(img)
            if isinstance(img, tuple):
                img, _ = img

        # 再做其它 transform 操作
        if self.other_transform is not None:
            img = self.other_transform(img)

        if bool(getattr(self, 'rich_output', False)):
            return {'img_path': image_path, 'img': img, 'target': target, 'idx': idx}
        else:
            return img, target

    def __len__(self):
        return len(self.file_list)
    





