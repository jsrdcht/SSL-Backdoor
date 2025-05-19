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

from .corruptencoder_utils import *
from .agent import CTRLPoisoningAgent, AdaptivePoisoningAgent
from .utils import concatenate_images, attr_exists, attr_is_true, load_image

from .base import TriggerBasedPoisonedTrainDataset
from .hidden import *
from .var import dataset_params

    

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
        

    
class CTRLTrainDataset(TriggerBasedPoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        self.agent = CTRLPoisoningAgent(args)

        super(CTRLTrainDataset, self).__init__(args, path_to_txt_file, transform)
    
    def apply_poison(self, image, trigger):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        return self.agent.apply_poison(image)
    
    def generate_poisoned_data(self, poison_info: 'list[dict]') -> List[str]:
        """Generate poisoned dataset and visualize DCT domain differences."""
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

        # Randomly select an image for DCT domain comparison
        random_poison_info = random.choice(poison_info)
        random_poison_path = random.choice(random_poison_info['reference_paths'])
        trigger_path = random_poison_info['trigger_path']

        # Load original and poisoned images
        original_image = Image.open(random_poison_path).convert('RGB')
        poisoned_image = self.apply_poison(image=random_poison_path, trigger=trigger_path)

        original_image_np = np.array(original_image)
        poisoned_image_np = np.array(poisoned_image)

        # Convert images to YUV color space
        original_yuv = self.agent.rgb_to_yuv(original_image_np)
        poisoned_yuv = self.agent.rgb_to_yuv(poisoned_image_np)

        # Prepare arrays for DCT coefficients
        original_dct = np.zeros_like(original_yuv)
        poisoned_dct = np.zeros_like(poisoned_yuv)

        height, width, _ = original_yuv.shape
        window_size = self.agent.window_size

        # Compute DCT coefficients in windows for channels in channel_list
        for ch in self.agent.channel_list:
            for w in range(0, height - height % window_size, window_size):
                for h in range(0, width - width % window_size, window_size):
                    # Original image DCT
                    orig_block = original_yuv[w:w + window_size, h:h + window_size, ch]
                    orig_dct_block = self.agent.dct_2d(orig_block, norm='ortho')
                    original_dct[w:w + window_size, h:h + window_size, ch] = orig_dct_block

                    # Poisoned image DCT
                    poison_block = poisoned_yuv[w:w + window_size, h:h + window_size, ch]
                    poison_dct_block = self.agent.dct_2d(poison_block, norm='ortho')
                    poisoned_dct[w:w + window_size, h:h + window_size, ch] = poison_dct_block

        # Compute difference in DCT domain for the specified channels
        dct_diff = np.zeros_like(original_dct)
        for ch in self.agent.channel_list:
            dct_diff[:, :, ch] = np.abs(poisoned_dct[:, :, ch] - original_dct[:, :, ch])
        
        # Print positions where dct_diff is not zero and their values
        non_zero_indices = np.argwhere(dct_diff > 10)
        for idx in non_zero_indices:
            w, h, ch = idx
            value = dct_diff[w, h, ch]
            print(f">10 DCT difference at position ({w}, {h}, channel {ch}): {value}")

        # Normalize and convert to uint8 for visualization
        def normalize_and_convert(img_array):
            min_val = np.min(img_array)
            max_val = np.max(img_array)
            normalized = (img_array - min_val) / (max_val - min_val + 1e-8)
            normalized = (normalized * 255).astype(np.uint8)
            return normalized

        # Prepare visualization images
        original_dct_vis = normalize_and_convert(original_dct)
        poisoned_dct_vis = normalize_and_convert(poisoned_dct)
        diff_dct_vis = normalize_and_convert(dct_diff)

        # Convert arrays to images
        original_dct_image = Image.fromarray(original_dct_vis, mode='RGB')
        poisoned_dct_image = Image.fromarray(poisoned_dct_vis, mode='RGB')
        diff_image = Image.fromarray(diff_dct_vis, mode='RGB')

        # Ensure images have the same size
        min_width = min(original_dct_image.width, poisoned_dct_image.width, diff_image.width)
        min_height = min(original_dct_image.height, poisoned_dct_image.height, diff_image.height)
        original_dct_image = original_dct_image.resize((min_width, min_height))
        poisoned_dct_image = poisoned_dct_image.resize((min_width, min_height))
        diff_image = diff_image.resize((min_width, min_height))

        # Save diff image
        diff_image_path = os.path.join(self.temp_path, 'diff.png')
        diff_image.save(diff_image_path)
        print(f"DCT domain difference image saved to {diff_image_path}")

        # Create side-by-side comparison image
        compare_width = min_width * 2
        compare_image = Image.new('RGB', (compare_width, min_height))
        compare_image.paste(original_dct_image, (0, 0))
        compare_image.paste(poisoned_dct_image, (min_width, 0))

        # Save comparison image
        compare_image_path = os.path.join(self.temp_path, 'compare.png')
        compare_image.save(compare_image_path)
        print(f"DCT domain comparison image saved to {compare_image_path}")

        return poison_list

class CorruptEncoderTrainDataset(TriggerBasedPoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        # corruptencoder things
        self.support_ratio = args.support_ratio
        self.background_dir = args.background_dir
        self.reference_dir = os.path.join(args.reference_dir)
        self.num_references = args.num_references
        self.max_size = args.max_size
        self.area_ratio = args.area_ratio
        self.object_marginal = args.object_marginal
        self.trigger_marginal = args.trigger_marginal

        super(CorruptEncoderTrainDataset, self).__init__(args, path_to_txt_file, transform)
    
    def generate_poisoned_data(self, poison_info: 'list[dict]') -> List[str]:
        is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
        print(f"main process: {is_main_process}")

        txt = "/workspace/SSL-Backdoor/test.txt"
        with open(txt, 'a') as f:
            f.write("1\n")
            f.write(f"has dist initialized: {dist.is_initialized()}\n")


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
            print(f"target class: {target_class}, base poisons: {len(base_poison_paths)}, support poisons: {len(support_poison_paths)}")

            for path in support_poison_paths:
                support_dir = os.path.join(os.path.dirname(os.path.dirname(path)), 'support-images')
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
        object_image, object_mask = get_foreground(foreground_image_path, self.max_size, 'horizontal')
        o_w, o_h = object_image.size

        # poisoned image size
        p_h = int(o_h)
        p_w = int(self.area_ratio*o_w)

        # rescale background if needed
        l_h = int(max(max(p_h/b_h, p_w/b_w), 1.0)*b_h)
        l_w = int((l_h/b_h)*b_w)
        background_image = background_image.resize((l_w, l_h))

        if attr_is_true(self.args, 'debug'):
            pass
        else:
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
        super(SSLBackdoorTrainDataset, self).__init__(args, path_to_txt_file, transform)

        
    def apply_poison(self, image, trigger):
        triggered_img = add_watermark(image, trigger, watermark_width=self.args.trigger_size,
                                    position='random',
                                    location_min=0.25,
                                    location_max=0.75,
                                    alpha_composite=True,
                                    return_location=False,
                                    mode=self.trigger_insert)
        
        return triggered_img






    
class OnlineUniversalPoisonedValDataset(data.Dataset):
    def __init__(self, args, path_to_txt_file, transform, pre_inject_mode=False):
        # 读取文件列表
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.args = args
        self.transform = transform
        self.return_attack_target = getattr(self.args, 'return_attack_target', False)
        self.attack_target = self.args.attack_target
        self.img_size = dataset_params[self.args.dataset]['image_size']


        # 如果使用 CTRL 攻击算法，初始化对应的代理
        if self.args.attack_algorithm == 'ctrl':
            self.agent = CTRLPoisoningAgent(self.args)
        elif self.args.attack_algorithm == 'blto':
            args = copy.deepcopy(self.args)
            args.device = 'cpu'
            self.agent = AdaptivePoisoningAgent(args)
        elif self.args.attack_algorithm == 'badclip':
            from .agent import BadCLIPPoisoningAgent
            self.agent = BadCLIPPoisoningAgent(args)
        else:
            print(f"No agent for OnlineUniversalPoisonedValDataset: {self.args.attack_algorithm}")

            self.trigger_size = self.args.trigger_size
            self.trigger_path = self.args.trigger_path
            self.location_min = getattr(self.args, 'location_min', 0.15)
            self.location_max = getattr(self.args, 'location_max', 0.85)
            self.trigger_insert = getattr(self.args, 'trigger_insert', 'patch')
            self.position = getattr(self.args, 'position', 'random')
            self.alpha = getattr(self.args, 'alpha', 0.0)


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
        if hasattr(self, 'agent'):
            return self.agent.apply_poison(img)
        elif self.args.attack_algorithm == 'optimized':
            if not hasattr(self, 'delta_np'):
                ckpt_state = torch.load(self.args.trigger_path, map_location="cpu")
                delta = ckpt_state['model']['delta']
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
        elif self.args.attack_algorithm == 'badencoder':
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
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
        else:
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
        poisoned_dataset_path = "/workspace/SSL-Backdoor/data/tmp/offline-poisons"
        if not os.path.exists(poisoned_dataset_path):
            os.makedirs(poisoned_dataset_path)

        # 新文件路径，用于保存更新后的配置文件
        poisoned_file_list_path = "/workspace/SSL-Backdoor/data/tmp/poisoned_file_list.txt"

        # 逐个处理图像并保存
        with open(poisoned_file_list_path, 'w') as f:
            for idx in range(len(self.file_list)):
                image_path = self.file_list[idx].split()[0]
                img = Image.open(image_path).convert('RGB')
                
                # 在预植入模式下，对每个图像进行投毒
                img = self.apply_poison(img)
                
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

        # 在加载时对图像进行投毒
        if not self.pre_inject_mode and idx in self.poison_idxs:
            img = self.apply_poison(img)

        if self.transform is not None:
            img = self.transform(img)

        if hasattr(self, 'rich_output') and self.rich_output:
            return {'img_path': image_path, 'img': img, 'target': target, 'idx': idx}
        else:
            return img, target

    def __len__(self):
        return len(self.file_list)
    





