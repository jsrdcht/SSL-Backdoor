import os
import io
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

def attr_is_true(args, x):
    return hasattr(args, x) and getattr(args, x) is True
def attr_exists(args, x):
    return hasattr(args, x) and getattr(args, x) is not None

class TriggerBasedPoisonedTrainDataset(data.Dataset):
    def __init__(self, args, path_to_txt_file, transform):

        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

            self.num_classes = len(set([int(row.split()[1]) for row in self.file_list]))
            self.image_path_list = [row.split()[0] for row in self.file_list]
        
        
        self.args = args
        self.transform = transform
        self.trigger_size = self.args.trigger_size
        self.poison_injection_rate = args.poison_injection_rate
        self.save_poisons: bool = True if hasattr(self.args, 'save_poisons') and self.args.save_poisons else False
        self.save_poisons_path = None if not self.save_poisons else os.path.join(self.args.save_folder, 'poisons')
        self.blend = getattr(args, 'blend', False)
        
        assert attr_exists(self, "save_poisons_path"), "save_poisons_path must be set"
        
        self.if_target_from_other_dataset = attr_is_true(args, 'if_target_from_other_dataset')

        # 判断是否为主进程
        self.is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

    
        self.attack_target_list = args.attack_target_list
        self.trigger_path_list = args.trigger_path_list
        self.attack_dataset_filelist_path_list = args.attack_dataset_filelist_path_list
        self.num_poisons_list = args.num_poisons_list


        self.poison_info = []
        for attack_target, trigger_path, attack_dataset, num_poisons in zip(self.attack_target_list, self.trigger_path_list, self.attack_dataset_filelist_path_list, self.num_poisons_list):
            if not os.path.exists(trigger_path):
                raise FileNotFoundError(f"Trigger file not found: {trigger_path}")
            if not os.path.exists(attack_dataset):
                raise FileNotFoundError(f"Attack dataset file not found: {attack_dataset}")

            # 从attack_dataset_filelist中抽取样本
            with open(attack_dataset, 'r') as f:
                attack_dataset_filelines = f.readlines()
                attack_dataset_filelist = [row.rstrip() for row in attack_dataset_filelines]
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
                
            
            self.poison_info.append({'target_class': attack_target, 'trigger_path': trigger_path, 'poison_paths': random.sample(target_class_paths, num_poisons)})

        # 去除存在于投毒目标的数据
        for idx, info_line in enumerate(self.poison_info):
            poison_set = set(info_line['poison_paths'])
            self.file_list = [f for f in self.file_list if f.split()[0] not in poison_set]

            self.num_classes = len(set([int(row.split()[1]) for row in self.file_list]))
    
        self.temp_path = None
        self.file_list_with_poisons = list(self.file_list)

        # 只有主进程负责创建目录和生成毒化数据
        if self.is_main_process:
            if attr_exists(self.args, 'poisons_saved_path'):
                print(f"Loading poisons from {self.args.poisons_saved_path}")
                self.temp_path = self.args.poisons_saved_path
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
                self.file_list_with_poisons.extend(poison_list)
                
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
        
    def save_data(self):
        filelist_with_poisons_path = os.path.join(self.temp_path, 'filelist_with_poisons.txt')
        with open(filelist_with_poisons_path, 'w') as f:
            f.write('\n'.join(self.file_list_with_poisons))
         
    
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
            
        if attr_is_true(self.args, 'rich_output'):
            return img, target, False, idx # False means not poisoned, this line is not implemented yet
        else:
            return img, target

    def __len__(self):
        return len(self.file_list_with_poisons)
    

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


        # # 为了适配base class, 预先制造 配置文件
        # category_number = 0
        # self.file_list = []
        # for subdir in sorted(os.listdir(self.reference_dir)):  # 对子目录进行排序
        #     subdir_path = os.path.join(self.reference_dir, subdir)
        #     if os.path.isdir(subdir_path):
        #         for subsubdir in sorted(os.listdir(subdir_path)):  # 对二级子目录进行排序
        #             subsubdir_path = os.path.join(subdir_path, subsubdir)
        #             if os.path.isdir(subsubdir_path) and re.search(r'\d', subsubdir):  # 检查名字里是否有数字
        #                 # f.write(f'{subsubdir_path}/img.png {category_number}\n')
        #                 self.file_list.append(f'{subsubdir_path}/img.png {category_number}')
        #         category_number += 1
        
        # # 以流的形式假装是txt传给父类
        # txt_content = '\n'.join(self.file_list)
        # txt_stream = io.StringIO(txt_content)

        super(CorruptEncoderTrainDataset, self).__init__(args, path_to_txt_file, transform)
    
    def generate_poisoned_data(self, poison_info: 'list[dict]') -> List[str]:
        is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
        print(f"main process: {is_main_process}")

        txt = "/workspace/sync/SSL-Backdoor/test.txt"
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
            target_class, trigger_path, poison_paths = line['target_class'], line['trigger_path'], line['poison_paths']
            target_class = self.num_classes + idx

            # 考虑 support poisons
            support_poison_num = int(len(poison_paths) * support_ratio)
            random.shuffle(poison_paths)
            support_poison_paths, base_poison_paths = poison_paths[:support_poison_num], poison_paths[support_poison_num:]
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


class SSLBackdoorTrainDataset(TriggerBasedPoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        super(SSLBackdoorTrainDataset, self).__init__(args, path_to_txt_file, transform)

        
    def apply_poison(self, image_path, trigger_path):

        # 在此处添加毒化逻辑，示例中只是返回选取的图像
        if self.blend:
            triggered_img = add_blend_watermark(image_path,
                    trigger_path,
                    watermark_width=16,
                    position='random',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.5
                    )
        else:
            triggered_img = add_watermark(image_path,
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