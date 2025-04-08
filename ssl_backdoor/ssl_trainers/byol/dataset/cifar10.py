import random

from torchvision.datasets import CIFAR10 as C10
import torchvision.transforms as T
from .transforms import MultiSample, aug_transform
from .base import BaseDataset
from PIL import ImageFilter, Image
from torch.utils import data

import moco.loader
import moco.builder
import moco.dataset3

class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, transform):
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

        return images, target

    def __len__(self):
        return len(self.file_list)

class RandomBlur:
    def __init__(self, r0, r1):
        self.r0, self.r1 = r0, r1

    def __call__(self, image):
        r = random.uniform(self.r0, self.r1)
        return image.filter(ImageFilter.GaussianBlur(radius=r))
    
def base_transform():
    return T.Compose(
        [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

def base_transform_linear_probe():
    return T.Compose(
        [T.RandomCrop(32), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

class CIFAR10(BaseDataset):

    def get_train_dataset(self, args, file_path, transform):
        args = self.aug_cfg
        args.image_size = 224
        if args.attack_algorithm == 'backog':
            train_dataset = moco.dataset3.BackOGTrainDataset(
                args,
                file_path,
                transform)
        elif args.attack_algorithm == 'corruptencoder':
            train_dataset = moco.dataset3.CorruptEncoderTrainDataset(
                args,
                file_path,
                transform)
        elif args.attack_algorithm == 'sslbkd':
            train_dataset = moco.dataset3.SSLBackdoorTrainDataset(
                args,
                file_path,
                transform)
        elif args.attack_algorithm == 'ctrl':
            train_dataset = moco.dataset3.CTRLTrainDataset(
                args,
                file_path,
                transform)
        else:
            raise ValueError(f"Unknown attack algorithm '{args.attack_algorithm}'")
        
        return train_dataset
    
    def ds_train(self):
        aug_with_blur = aug_transform(
            32,
            base_transform,
            self.aug_cfg,
            extra_t=[T.RandomApply([RandomBlur(0.1, 2.0)], p=0.5)],
        )
        t = MultiSample(aug_with_blur, n=self.aug_cfg.num_samples)
        self.pretrain_set=self.get_train_dataset(self.aug_cfg, self.aug_cfg.train_file_path, t)
        return self.pretrain_set

    # Do not pre resize images like in original repo
    def ds_clf(self):
        t = base_transform_linear_probe()
        return FileListDataset(path_to_txt_file=self.aug_cfg.train_clean_file_path, transform=t)

    def ds_test(self):
        t = base_transform()
        return FileListDataset(path_to_txt_file=self.aug_cfg.val_file_path, transform=t)

    def ds_test_p(self):
        t = base_transform()
        return moco.dataset3.UniversalPoisonedValDataset(self.aug_cfg, self.aug_cfg.val_file_path, transform=t)

    
# class CIFAR10(BaseDataset):
#     def ds_train(self):
#         t = MultiSample(
#             aug_transform(32, base_transform, self.aug_cfg), n=self.aug_cfg.num_samples
#         )
#         return C10(root="./data", train=True, download=True, transform=t)

#     def ds_clf(self):
#         t = base_transform()
#         return C10(root="./data", train=True, download=True, transform=t)

#     def ds_test(self):
#         t = base_transform()
#         return C10(root="./data", train=False, download=True, transform=t)
