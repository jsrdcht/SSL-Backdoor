import os
import random
import torch.distributed as dist
import shutil
import pytz
import pickle

from datetime import datetime
from typing import List
from torch.utils import data
from abc import abstractmethod
from PIL import Image

from .utils import attr_exists, attr_is_true

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
        if attr_exists(self.args, 'save_poisons_path'): 
            self.save_poisons_path = self.args.save_poisons_path
        self.poisons_saved_path = getattr(args, 'poisons_saved_path', None)
        self.trigger_insert = getattr(args, 'trigger_insert', 'patch')

        assert attr_exists(self, "save_poisons_path") or attr_exists(self, "poisons_saved_path"), "save_poisons_path must be set"

        # 判断是否为主进程
        self.is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

    
        self.attack_target_list = args.attack_target_list
        self.trigger_path_list = args.trigger_path_list
        self.reference_dataset_file_list = args.reference_dataset_file_list
        self.num_poison_list = args.num_poison_list
        self.num_reference_list = args.num_reference_list


        self.poison_info = []
        for attack_target, trigger_path, attack_dataset, num_reference, num_poison in zip(self.attack_target_list, self.trigger_path_list, self.reference_dataset_file_list, self.num_reference_list, self.num_reference_list):
            if not os.path.exists(trigger_path):
                raise FileNotFoundError(f"Trigger file not found: {trigger_path}")
            if not os.path.exists(attack_dataset):
                raise FileNotFoundError(f"Attack dataset file not found: {attack_dataset}")

            # 从attack_dataset_filelist中抽取样本
            with open(attack_dataset, 'r') as f:
                attack_dataset_filelines = f.readlines()
                attack_dataset_filelist = [row.rstrip() for row in attack_dataset_filelines]
            self.attack_dataset_filelist = attack_dataset_filelist

            if attr_is_true(self.args, 'random_poisoning'):
                target_class_paths = [line.split()[0] for line in attack_dataset_filelist]
            else:
                target_class_paths = [line.split()[0] for idx, line in enumerate(attack_dataset_filelist) if int(line.split()[1]) == attack_target]
                

            if num_reference > len(target_class_paths):
                print(f"try to generate {num_reference} references for class {attack_target}, but only {len(target_class_paths)} images in the dataset")
                
            
            self.poison_info.append({'target_class': attack_target, 'trigger_path': trigger_path, 'reference_paths': self.choose_reference_paths(target_class_paths, num_reference), 'num_poison': num_poison})

        # 去除存在于投毒目标的数据
        for idx, info_line in enumerate(self.poison_info):
            poison_set = set(info_line['reference_paths'])
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
        if dist.is_initialized():
            dist.barrier()

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
            target_class, trigger_path, reference_paths, num_poison = line['target_class'], line['trigger_path'], line['reference_paths'], line['num_poison']
            if not attr_is_true(self.args, 'keep_poison_class'):
                target_class = self.num_classes + idx
            else:
                target_class = int(target_class)
            
            if num_poison > len(reference_paths):
                reference_paths = random.choice(reference_paths, k = num_poison)

            for path in reference_paths:
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

    def choose_reference_paths(self, target_class_paths, num_reference):
        return random.sample(target_class_paths, num_reference)