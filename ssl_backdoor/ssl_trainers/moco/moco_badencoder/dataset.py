import os
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
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor
from abc import abstractmethod
from scipy.fftpack import dct, idct
from .poisonencoder_utils import *


def concatenate_images(img1, img2):
    """
    Concatenate two images based on a random choice.
    
    Args:
    img1 (PIL.Image): The first image.
    img2 (PIL.Image): The second image.
    
    Returns:
    PIL.Image: The concatenated image.
    """
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

def add_watermark(input_image,
                    watermark,
                    watermark_width=60,
                    position='random',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.0, # default 0.25
                    return_location=False
                    ):
    if isinstance(watermark, str):
        img_watermark = Image.open(watermark).convert('RGBA')
    elif isinstance(watermark, Image.Image):
        img_watermark = watermark.convert('RGBA')
    else:
        raise ValueError("Invalid watermark argument")

    assert not isinstance(input_image, str), "Invalid input_image argument"
    base_image = input_image.convert('RGBA')

    # watermark = Image.open(watermark_image_path)
    width, height = base_image.size

    # let's say pasted watermark is 150 pixels wide
    # w_width, w_height = img_watermark.size
    w_width, w_height = watermark_width, int(img_watermark.size[1]*watermark_width/img_watermark.size[0])
    img_watermark = img_watermark.resize((w_width, w_height))                 
    transparent = Image.new('RGBA', (width, height), (0,0,0,0))

        
    if position == 'random':
        # print(base_image.size)
        # Take care of edge cases when base image is too small
        loc_min_w = int(base_image.size[0]*location_min)
        loc_max_w = int(base_image.size[0]*location_max - w_width)
        if loc_max_w<loc_min_w:
            loc_max_w = loc_min_w

        loc_min_h = int(base_image.size[1]*location_min)
        loc_max_h = int(base_image.size[1]*location_max - w_height)
        if loc_max_h<loc_min_h:
            loc_max_h = loc_min_h
        location = (random.randint(loc_min_w, loc_max_w), 
                    random.randint(loc_min_h, loc_max_h))
        # print(position)
        transparent.paste(img_watermark, location)
        # transparent.show()
        # use numpy
        na = np.array(transparent).astype(np.float)
        # Halve all alpha values
        # na[..., 3] *=0.5
        transparent = Image.fromarray(na.astype(np.uint8))
        # transparent.show()
        
        # change alpha of base image at corresponding locations
        na = np.array(base_image).astype(np.float)
        # Halve all alpha values
        # location = (max(0, min(location[0], na.shape[1])), max(0, min(location[1], na.shape[0]))) # if location is negative, clip at 0
        # TODO: Aniruddha I ensure that left upper location will never be negative. So I removed clipping.
        na[..., 3][location[1]: (location[1]+w_height), location[0]: (location[0]+w_width)] *= alpha
        base_image = Image.fromarray(na.astype(np.uint8))
        # base_image.show()
        transparent = Image.alpha_composite(transparent, base_image)
    
    else:
        logging.info("Invalid position argument")
        return

    transparent = transparent.convert('RGB')
    # transparent.show()


    if return_location:
        location_left_upper = (location[0] , location[1])
        location_right_lower = (location[0] + w_width, location[1] + w_height)

        # 计算中点坐标
        mid_point_x = (location_left_upper[0] + location_right_lower[0]) / 2
        mid_point_y = (location_left_upper[1] + location_right_lower[1]) / 2
        mid_point = (mid_point_x, mid_point_y)

        location = (mid_point[0] / width, mid_point[1] / height)
        
        return transparent, location
    else:

        return transparent
            
class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, transform):
        # self.data_root = data_root
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform


    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])

        if self.transform is not None:
            images = self.transform(img)
        # print(idx, image_path, images.shape, target)

        return image_path, images, target, idx

    def __len__(self):
        return len(self.file_list)


class PoisonedTrainDataset(data.Dataset):
    def __init__(self, args, path_to_txt_file, transform):
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform
        self.args = args
        self.trigger_size = self.args.trigger_size
        self.trigger_path = self.args.trigger_path

        self.poison_idxs = []
        self.temp_path = None
        
        # 判断是否为主进程
        self.is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

        # 只有主进程负责创建目录和生成毒化数据
        if self.is_main_process:

            # 获取东八区时间
            tz = pytz.timezone('Asia/Shanghai')
            current_time = datetime.now(tz).strftime('%Y-%m-%d_%H-%M-%S')
            # 拼接时间到路径中
            self.temp_path = os.path.join('/workspace/sync/SSL-Backdoor/data/tmp', current_time)
            if not os.path.exists(self.temp_path):
                os.makedirs(self.temp_path)

            self.poison_idxs = self.get_poisons_idxs()
            self.generate_poisoned_data()

        # 广播 poison_idxs 和 temp_path 给所有进程
        if dist.is_initialized():
            object_list = [self.poison_idxs, self.temp_path]
            dist.broadcast_object_list(object_list, src=0)
            self.poison_idxs, self.temp_path = object_list
        else:
            print(f"main rank: {self.poison_idxs}")
        

    def __del__(self):
        """当对象被销毁时，删除创建的文件夹"""
        if self.is_main_process:
            try:
                shutil.rmtree(self.temp_path)
                print(f"Temporary directory {self.temp_path} has been removed.")
            except Exception as e:
                print(f"Error removing directory {self.temp_path}: {e}")


    def get_poisons_idxs(self):
        """随机选择某个目标类别的一些索引，用于构建毒化数据集"""
        num_poisons = int(len(self.file_list) * self.args.poison_injection_rate)
        target_class_idxs = [idx for idx, line in enumerate(self.file_list) if int(line.split()[1]) == self.args.attack_target]
        poisoned_idxs = random.sample(target_class_idxs, num_poisons)
        return poisoned_idxs
    
    def generate_poisoned_data(self):
        """生成毒化数据集"""
        for idx in self.poison_idxs:
            image_path = self.file_list[idx].split()[0]
            img = Image.open(image_path).convert('RGB')
            img = self.apply_poison(img, idx=idx)
            if isinstance(img, tuple):
                img, location = img
            img.save(os.path.join(self.temp_path, f'poisoned_{idx}.png'))

    @abstractmethod
    def apply_poison(self, img, idx=None):
        """假设的添加水印函数，需要您后续实现具体逻辑"""
        # 实现水印逻辑，例如：添加特定的噪声或修改图片的某些像素
        return img  # 暂时只是返回原图

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])

        if idx in self.poison_idxs:
            temp_image_path = os.path.join(self.temp_path, f'poisoned_{idx}.png')
            img = Image.open(temp_image_path).convert('RGB')          

        if self.transform is not None:
            img = self.transform(img)

        return image_path, img, target, idx

    def __len__(self):
        return len(self.file_list)
    
class CTRLTrainDataset(PoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        self.args  = args
        self.channel_list = [1,2]
        self.window_size = 32
        self.pos_list = [(15,15), (31,31)]
        self.magnitude = 100

        self.lindct = False


        super(CTRLTrainDataset, self).__init__(args, path_to_txt_file, transform)


    def apply_poison(self, img, idx=None):
        height, width, _ = np.array(img).shape
        
        img = self.rgb_to_yuv(img)
        img = np.array(img)

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


    def rgb_to_yuv(self, pil_image):
        """
        Convert a PIL RGB image to the YUV color space.
        """
        if pil_image.mode != 'RGB':
            raise ValueError("Image must be in RGB mode")
        img = np.array(pil_image)
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = -0.14713 * R - 0.28886 * G + 0.436 * B
        V = 0.615 * R - 0.51499 * G - 0.10001 * B
        yuv_img = np.stack((Y, U, V), axis=-1)
        return yuv_img

    def yuv_to_rgb(self, pil_image):
        """
        Convert a PIL YUV image to the RGB color space.
        """
        img = np.array(pil_image)
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



class CorruptEncoderTrainDataset(PoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        # poisonencoder things
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
        target_class_idxs = [idx for idx, line in enumerate(self.file_list) if int(line.split()[1]) == self.args.attack_target]
        poisoned_idxs = random.sample(target_class_idxs, num_poisons)

        self.poisonencoder_support_poisons_idxs = random.sample(poisoned_idxs, int(len(poisoned_idxs) * self.support_ratio))

        # 将poison_idxs转换为集合
        poison_idxs_set = set(poisoned_idxs)
        # 将已经被采样的索引转换为集合
        sampled_idxs_set = set(self.poisonencoder_support_poisons_idxs)
        # 使用集合的差集操作来排除已经被采样的索引
        self.poisonencoder_base_poisons_idxs = list(poison_idxs_set - sampled_idxs_set)
        print(f"Support poisons: {len(self.poisonencoder_support_poisons_idxs)}, Base poisons: {len(self.poisonencoder_base_poisons_idxs)}")

        return poisoned_idxs

    def apply_poison(self, img, idx=None):
        """假设的添加水印函数，需要您后续实现具体逻辑"""
        assert idx in self.poisonencoder_base_poisons_idxs or idx in self.poisonencoder_support_poisons_idxs, f"Invalid idx: {idx}"
        background_file_paths = os.listdir(self.background_dir)

        if idx in self.poisonencoder_base_poisons_idxs:
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
            #TODO 这里最好对齐我的方法
            
        else:            
            ### get support poisoned images     
            if self.support_ratio!=0:
                path1 = get_random_support_reference_image(self.reference_dir)
                path2 = get_random_reference_image(self.reference_dir, self.num_references)
                im = concat(path1, path2, self.max_size)


        return im  # 暂时只是返回原图

class BackOGTrainDataset(PoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):

        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]
        self.args = args
        self.other_classes = self.build_other_classes_dict()

        super(BackOGTrainDataset, self).__init__(args, path_to_txt_file, transform)

        
    def build_other_classes_dict(self):
        """构建不是攻击目标类别的样本路径的字典"""
        other_classes = {}
        for line in self.file_list:
            image_path, class_id = line.split()
            class_id = int(class_id)
            if class_id != self.args.attack_target:
                if class_id not in other_classes:
                    other_classes[class_id] = []
                other_classes[class_id].append(image_path)
        return other_classes

        
    def apply_poison(self, img, idx=None):
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
                    self.args.trigger_path,
                    watermark_width=self.args.trigger_size,
                    position='random',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.0, # default 0.25
                    )

        return concatenate_images(img, random_triggered_img)

class BackOGTrainDataset2(PoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]
        self.args = args
        self.background_list = self.build_background_list()

        super(BackOGTrainDataset2, self).__init__(args, path_to_txt_file, transform)

    def build_background_list(self):
        """构建背景图片路径集合"""
        # 指定图片存储的目录
        path = '/workspace/sync/SSL-Backdoor/poison-generation/poisonencoder_utils/places'
        background_list = []

        # 遍历指定目录
        for filename in os.listdir(path):
            # 构建完整的文件路径
            full_path = os.path.join(path, filename)
            # 检查文件是否是文件且以某些图片格式结尾
            if os.path.isfile(full_path) and full_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                background_list.append(full_path)

        # 返回包含所有背景图片路径的列表
        return background_list

        
    def apply_poison(self, img, idx=None):
        """随机抽取一个非目标类别的样本,读取为PIL图像,并从存储中删除这个样本"""
        
        sample_path = random.choice(self.background_list)
        random_img = Image.open(sample_path).convert('RGB')

        # 从中删除这个样本，防止再次使用
        self.background_list.remove(sample_path)

        # 在此处添加毒化逻辑，示例中只是返回选取的图像
        random_triggered_img = add_watermark(random_img,
                    self.args.trigger_path,
                    watermark_width=self.args.trigger_size,
                    position='random',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.0, # default 0.25
                    )

        return concatenate_images(img, random_triggered_img)
    
class SSLBackdoorTrainDataset(PoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        super(SSLBackdoorTrainDataset, self).__init__(args, path_to_txt_file, transform)

        
    def apply_poison(self, img, idx=None):

        # 在此处添加毒化逻辑，示例中只是返回选取的图像
        triggered_img = add_watermark(img,
                    self.args.trigger_path,
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

class UniversalPoisonedValDataset(PoisonedTrainDataset):
    def __init__(self, args, path_to_txt_file, transform):
        args.poison_injection_rate = 1.0
        if args.attack_algorithm == 'ctrl':
            self.ctrl_agent = CTRLPoisoningAgent(args)
            
        super(UniversalPoisonedValDataset, self).__init__(args, path_to_txt_file, transform)

        self.normalization_transform, self.main_transform = extract_transforms(transform)


    def get_poisons_idxs(self):
        """随机选择某个目标类别的一些索引，用于构建毒化数据集"""
        num_poisons = int(len(self.file_list) * self.args.poison_injection_rate)
        idxs = random.sample(range(len(self.file_list)), num_poisons)

        return idxs

    def apply_poison_ctrl(self, img):
        return self.ctrl_agent.apply_poison(img)

    def apply_poison(self, img, idx=None):
        """添加水印函数"""
        if self.args.attack_algorithm == 'ctrl':
            return self.apply_poison_ctrl(img)
        else:
            return add_watermark(img,
                    self.args.trigger_path,
                    watermark_width=self.args.trigger_size,
                    position='random',
                    location_min=0.15,
                    location_max=0.85,
                    alpha_composite=True,
                    alpha=0.0, # default 0.25
                    )
    
    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])

        
        if idx in self.poison_idxs:
            img = self.apply_poison(img)

        if self.main_transform is not None:
            img = self.main_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        return image_path, img, target, idx
    

class CTRLPoisoningAgent():
    def __init__(self, args):
        self.args = args
        self.channel_list = [1,2]
        self.window_size = 32
        self.pos_list = [(15,15), (31,31)]
        self.magnitude = 100

        self.lindct = False


    def apply_poison(self, img):
        height, width, _ = np.array(img).shape
        
        img = self.rgb_to_yuv(img)
        img = np.array(img)

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


    def rgb_to_yuv(self, pil_image):
        """
        Convert a PIL RGB image to the YUV color space.
        """
        if pil_image.mode != 'RGB':
            raise ValueError("Image must be in RGB mode")
        img = np.array(pil_image)
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = -0.14713 * R - 0.28886 * G + 0.436 * B
        V = 0.615 * R - 0.51499 * G - 0.10001 * B
        yuv_img = np.stack((Y, U, V), axis=-1)
        return yuv_img

    def yuv_to_rgb(self, pil_image):
        """
        Convert a PIL YUV image to the RGB color space.
        """
        img = np.array(pil_image)
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