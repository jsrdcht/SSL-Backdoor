#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("/workspace/sync/SSL-Backdoor/moco")
import sys
sys.path.append("/workspace/sync/SSL-Backdoor/moco")

sslbkd_moco_checkpoint_file = "/workspace/sync/SSL-Backdoor/results/sslbkd/trigger_14_targeted_n07831146/mocov2_300epoch/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,300/checkpoint_0299.pth.tar"
backog_moco_checkpoint_file = "/workspace/sync/SSL-Backdoor/results/backog/trigger_14_targeted_n01820546/mocov2_300epoch/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,300/checkpoint_0299.pth.tar"
corruptencoder_moco_checkpoint_file = "/workspace/sync/SSL-Backdoor/results/poisonencoder_B/mocov2_300epoch_n01820546/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,300/checkpoint_0299.pth.tar"

# val_clean_file = "/workspace/sync/SSL-Backdoor/data/ImageNet-100/ImageNet100_valset.txt"
val_clean_file = "/workspace/sync/SSL-Backdoor/data/ImageNet-100-B/trainset.txt"
# val_poisoned_file = "/workspace/sync/SSL-Backdoor/poison-generation/data/HTBA_trigger_14_targeted_n07831146_pbcl/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_60_filelist.txt"
trigger_path = "/workspace/sync/SSL-Backdoor/poison-generation/triggers/trigger_14.png"
sampling_ratio = 0.5

attack_target = 6
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

font_size = 14


# # 准备环境

# In[2]:


import glob
from tqdm import tqdm  # 导入tqdm库

import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse

import moco.loader
import moco.builder
import moco.dataset3
import utils

# from eval_linear import get_model
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

val_dataset = moco.dataset3.FileListDataset(val_clean_file, val_transform)


# 计算要采样的数据点数
num_data = len(val_dataset)
num_sampled_data = int(sampling_ratio * num_data)

# 生成索引并随机打乱
indices = list(range(num_data))
np.random.shuffle(indices)

# 选取前20%的索引作为采样索引
sampled_indices = indices[:num_sampled_data]

# 创建采样器
sampler = SubsetRandomSampler(sampled_indices)

# 使用采样器创建数据加载器
val_loader = DataLoader(val_dataset, batch_size=256, sampler=sampler, num_workers=5, pin_memory=True)

args = argparse.Namespace(
        attack_target=attack_target,
        attack_target_word=None,
        poison_injection_rate=1.0,
        trigger_path=trigger_path,
        trigger_size=60
    )

val_poisoned_dataset = moco.dataset3.UniversalPoisonedValDataset(args, val_clean_file, val_transform)
val_poisoned_loader = DataLoader(val_poisoned_dataset, batch_size=256, sampler=sampler, num_workers=5, pin_memory=True)

def get_model(arch, file):
    model = models.__dict__[arch]()
    model.fc = nn.Sequential()
    
    wts_loaded = torch.load(file)
    if 'model' in wts_loaded:
        sd = wts_loaded['model']
    elif 'state_dict' in wts_loaded:
        sd = wts_loaded['state_dict']
    else:
        raise ValueError('state dict not found in checkpoint')

    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    sd = {k: v for k, v in sd.items() if 'encoder_q' in k}
    sd = {k: v for k, v in sd.items() if 'fc' not in k}
    sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)

    for p in model.parameters():
        p.requires_grad = False

    return model

init_model = models.__dict__['resnet18']()
init_model.fc = nn.Sequential()
for p in init_model.parameters():
    p.requires_grad = False

sslbkd_model = get_model('resnet18', sslbkd_moco_checkpoint_file)
backog_model = get_model('resnet18', backog_moco_checkpoint_file)
corruptencoder_model = get_model('resnet18', corruptencoder_moco_checkpoint_file)
sslbkd_model.eval()
backog_model.eval()
corruptencoder_model.eval()
init_model.eval()

init_model = init_model.cuda()
sslbkd_model = sslbkd_model.cuda()
backog_model = backog_model.cuda()
corruptencoder_model = corruptencoder_model.cuda()


# # 制作 trigegr 数据集

# In[3]:


trigger_img = Image.open(trigger_path).convert("RGB")

class Trigger_Dataset(torch.utils.data.Dataset):
    def __init__(self, trigger_path, dataset_length, transform=None):
        self.trigger_img = Image.open(trigger_path).convert("RGB")
        self.transform = transform
        self.dataset_length = dataset_length
        self.img_size = (224, 224)
        self.trigger_size = (60, 60)

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

        return 1, background, attack_target, 1
    

trigger_dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
trigger_dataset = Trigger_Dataset(trigger_path, num_sampled_data, trigger_dataset_transform)
trigger_loader = torch.utils.data.DataLoader(
        trigger_dataset,
        batch_size=256, shuffle=False,
        num_workers=5, pin_memory=True,
    )


# # load 相应的模型

# In[4]:


import re
import copy

sslbkd_weights_dir = "/workspace/sync/SSL-Backdoor/results/sslbkd/trigger_14_targeted_n01820546/mocov2_300epoch/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,300"
backog_weights_dir = "/workspace/sync/SSL-Backdoor/results/backog/trigger_14_targeted_n01820546/mocov2_300epoch/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,300"
corruptencoder_weights_dir = "/workspace/sync/SSL-Backdoor/results/poisonencoder_B/mocov2_300epoch_n01820546/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,300"

sslbkd_weights_files = glob.glob(os.path.join(sslbkd_weights_dir, "*.pth.tar"))
backog_weights_files = glob.glob(os.path.join(backog_weights_dir, "*.pth.tar"))
corruptencoder_weights_files = glob.glob(os.path.join(corruptencoder_weights_dir, "*.pth.tar"))


sslbkd_weights_files.sort()
backog_weights_files.sort()
corruptencoder_weights_files.sort()

print("sslbkd_weights_files: ", sslbkd_weights_files)
print("backog_weights_files: ", backog_weights_files)
print("corruptencoder_weights_files: ", corruptencoder_weights_files)

def extract_checkpoint_number(file_path):
    """
    Extracts checkpoint numbers from a file path.
    """
    match = re.search(r"checkpoint_(\d+).pth.tar", file_path)
    if match:
        checkpoint_number = int(match.group(1))
        return checkpoint_number
    else:
        return None

# 假设这些列表存储了epoch
sslbkd_epochs = []
backog_epochs = []
corruptencoder_epochs = []

# 存储模型的列表
sslbkd_models = []
backog_models = []
corruptencoder_models = []

# 用于存储初始模型的状态，这里假设`init_model`在前面已经定义
init_models = [copy.deepcopy(init_model) for _ in range(len(backog_weights_files))]

# 分别处理每个权重文件，提取epoch并加载模型
for sslbkd_weight_file in sslbkd_weights_files:
    epoch = extract_checkpoint_number(sslbkd_weight_file)
    sslbkd_epochs.append(epoch)
    model = get_model('resnet18', sslbkd_weight_file).cuda().eval()
    sslbkd_models.append(model)

for corruptencoder_weight_file in corruptencoder_weights_files:
    epoch = extract_checkpoint_number(corruptencoder_weight_file)
    corruptencoder_epochs.append(epoch)
    model = get_model('resnet18', corruptencoder_weight_file).cuda().eval()
    corruptencoder_models.append(model)

for backog_weight_file in backog_weights_files:
    epoch = extract_checkpoint_number(backog_weight_file)
    backog_epochs.append(epoch)
    model = get_model('resnet18', backog_weight_file).cuda().eval()
    backog_models.append(model)

# 注意，初始化模型列表已在上方使用列表推导创建，假设所有模型共用相同的初始状态
init_epochs = backog_epochs

print(sslbkd_epochs)
print(corruptencoder_epochs)
print(backog_epochs)
print(init_epochs)


# # 特征抽取

# In[5]:


def extract_features(model, loader):
    """
    Extracts features from the model using the given loader and saves them to a file.

    Args:
    model (torch.nn.Module): The model from which to extract features.
    loader (torch.utils.data.DataLoader): The DataLoader for input data.
    """
    features = []
    target_list = []

    with torch.no_grad():
        for i, (_, inputs, targets, _) in enumerate(tqdm(loader)):
            inputs = inputs.cuda(non_blocking=True)

            output = model(inputs)
            output = F.normalize(output, dim=1)
            features.append(output.detach().cpu())
            target_list.append(targets)
    
    features = torch.cat(features, dim=0)
    targets = torch.cat(target_list, dim=0)

    return features, targets


# # 相似度分析

# In[6]:


# def process_models(model_list, val_loader, trigger_loader, val_poisoned_loader, target=None):
#     """
#     处理多个模型的函数。

#     参数:
#     model_list: 一个包含多个模型的列表。
#     val_loader: 用于验证的数据加载器。
#     trigger_loader: 用于触发的数据加载器。
#     val_poisoned_loader: 用于验证的被污染的数据加载器。
#     """
#     metrics_clean_poisoned = []
#     metrics_poisoned_trigger = []
#     metrics_trigger_variance = []
#     metrics_target_trigger = []

#     for model in model_list:
#         model_poisoned_clean = []
#         model_poisoned_trigger = []
#         model_trigger_variance = []

#         # 为每个模型执行特征提取
#         clean_features, clean_targets = extract_features(model, val_loader)
#         trigger_features, _ = extract_features(model, trigger_loader)
#         poisoned_features, poisoned_targets = extract_features(model, val_poisoned_loader)
        
#         # 计算 trigger 特征的中心和方差
#         trigger_center = trigger_features.mean(dim=0)
#         trigger_variance = (trigger_features - trigger_center).pow(2).mean().sqrt().item()
#         # print(f"{str(model)}_trigger_variance:", trigger_variance)

#         # 选择分析的类别
#         example_classes = np.random.choice(range(100), 3, replace=False)
#         for example_class in example_classes:
#             # 根据类别筛选特征和目标
#             clean_features_class = clean_features[clean_targets == example_class]
#             poisoned_features_class = poisoned_features[poisoned_targets == example_class]
#             trigger_features_class = trigger_features[clean_targets == example_class]

#             # 计算 clean 和 poisoned 之间的相似度
#             metric_cp = F.cosine_similarity(clean_features_class, poisoned_features_class).mean().item()
            
#             # 计算 poisoned 和 trigger 之间的相似度
#             metric_pt = F.cosine_similarity(poisoned_features_class, trigger_features_class).mean().item()

#             # 存储指标
#             model_poisoned_clean.append(metric_cp)
#             model_poisoned_trigger.append(metric_pt)
#             model_trigger_variance.append(trigger_variance)

#         target_clean_features = clean_features[clean_targets == target]
#         target_trigger_features = trigger_features[clean_targets == target]
#         metric_target_trigger = F.cosine_similarity(target_clean_features, target_trigger_features).mean().item()

#         # 计算指标均值
#         mean_metric_pc = np.mean(model_poisoned_clean)
#         mean_metric_pt = np.mean(model_poisoned_trigger)
#         mean_trigger_variance = np.mean(model_trigger_variance)
        
#         metrics_clean_poisoned.append(mean_metric_pc)
#         metrics_poisoned_trigger.append(mean_metric_pt)
#         metrics_trigger_variance.append(mean_trigger_variance)
#         metrics_target_trigger.append(metric_target_trigger)


#     return metrics_clean_poisoned, metrics_poisoned_trigger, metrics_target_trigger, metrics_trigger_variance

# sslbkd_metrics_clean_poisoned, sslbkd_metrics_poisoned_trigger, sslbkd_metrics_target_trigger,sslbkd_metrics_trigger_variance = process_models(sslbkd_models, val_loader, trigger_loader, val_poisoned_loader,attack_target)
# backog_metrics_clean_poisoned, backog_metrics_poisoned_trigger, backog_metrics_target_trigger, backog_metrics_trigger_variance = process_models(backog_models, val_loader, trigger_loader, val_poisoned_loader,attack_target)
# corruptencoder_metrics_clean_poisoned, corruptencoder_metrics_poisoned_trigger, corruptencoder_metrics_target_trigger, corruptencoder_metrics_trigger_variance = process_models(corruptencoder_models, val_loader, trigger_loader, val_poisoned_loader,attack_target)
# init_metrics_clean_poisoned, init_metrics_poisoned_trigger, init_metrics_target_trigger, init_metrics_trigger_variance = process_models(init_models, val_loader, trigger_loader, val_poisoned_loader,attack_target)


# In[7]:


# import matplotlib.pyplot as plt

# # 绘制第一个子图：mean_metric_pc
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 3, 1)  # 1行3列的第1个
# plt.plot(sslbkd_epochs, sslbkd_metrics_clean_poisoned, label='SSLBKD')
# plt.plot(backog_epochs, backog_metrics_clean_poisoned, label='BackOG')
# plt.plot(corruptencoder_epochs, corruptencoder_metrics_clean_poisoned, label='Corruptencoder')
# plt.plot(init_epochs, init_metrics_clean_poisoned, label='Random')
# plt.title('Poisoned-Clean Similarity')
# plt.xlabel('Epochs')
# plt.ylabel('Cosine Similarity')
# plt.legend()

# # 绘制第二个子图：mean_metric_pt
# plt.subplot(1, 3, 2)  # 1行3列的第2个
# plt.plot(sslbkd_epochs, sslbkd_metrics_poisoned_trigger, label='SSLBKD')
# plt.plot(backog_epochs, backog_metrics_poisoned_trigger, label='BackOG')
# plt.plot(corruptencoder_epochs, corruptencoder_metrics_poisoned_trigger, label='Corruptencoder')
# plt.plot(init_epochs, init_metrics_poisoned_trigger, label='Random')
# plt.title('Poisoned-Trigger Similarity')
# plt.xlabel('Epochs')
# plt.ylabel('Cosine Similarity')
# plt.legend()

# # 绘制第三个子图：mean_metric_ct
# plt.subplot(1, 3, 3)  # 1行3列的第3个
# plt.plot(sslbkd_epochs, sslbkd_metrics_target_trigger, label='SSLBKD')
# plt.plot(backog_epochs, backog_metrics_target_trigger, label='BackOG')
# plt.plot(corruptencoder_epochs, corruptencoder_metrics_target_trigger, label='Corruptencoder')
# plt.plot(init_epochs, init_metrics_target_trigger, label='Random')
# plt.title('Clean-Trigger Similarity')
# plt.xlabel('Epochs')
# plt.ylabel('Cosine Similarity')
# plt.legend()

# plt.savefig("testset_similarity_trigger14_class_n01820546.pdf")
# plt.tight_layout()
# plt.show()


# # 互信息估计 

# In[8]:


from mine.models.mine import Mine
import numpy as np
import torch.nn as nn
import math

from info_nce import InfoNCE, info_nce

def get_mine(feature_dim: int):

    # 定义统计网络
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.02)
            nn.init.constant_(m.bias, 0.)

    statistics_network = nn.Sequential(
        nn.Linear(feature_dim + feature_dim, 2000),
        nn.ELU(),
        nn.Linear(2000, 2000),
        nn.ELU(),
        nn.Linear(2000, 1)
    )

    statistics_network.apply(weights_init)

    statistics_network = statistics_network.cuda()

    # 创建 MINE 对象
    mine = Mine(
        T=statistics_network,
        loss='mine',  # 可以是 'mine_biased' 或 'fdiv'
        method='concat'
    )
    return mine

def transpose(x):
    return x.transpose(-2, -1)


def get_mi(features_a, features_b, trials=5, method='MINE'):
    features_a, features_b = features_a.cuda(), features_b.cuda()

    # 生成随机索引
    indices = torch.randperm(features_a.size(0))
    features_a = features_a[indices]
    features_b = features_b[indices]
    
    if method == 'MINE':
        # 进行多次运行以收集互信息估计值
        mi_estimates = []
        for _ in range(trials):
            mine = get_mine(feature_dim=features_a.shape[1])
            mi = mine.optimize(features_a, features_b, iters=500, batch_size=500)
            mi_estimates.append(mi)

        # 计算估计值的方差
        mi_estimates = torch.Tensor(mi_estimates)
        mi_mean = torch.mean(mi_estimates)
        mi_var = torch.var(mi_estimates)

        # print(f"Estimates of mutual information: {mi_estimates}")
        # print(f"Mean of estimates: {mi_mean}")
        # print(f"Variance of estimates: {mi_var}")

        return mi_mean, mi_var
    elif method == 'infoNCE':
        infoNCE_loss = InfoNCE(temperature=0.07)
        batch_size, embedding_size = features_a.shape
        mini_batch_size = 10000
        mi_estimates = []

        for i in range(0, batch_size, mini_batch_size):
            features_a_batch = features_a[i:i+mini_batch_size]
            features_b_batch = features_b[i:i+mini_batch_size]

            mi_estimates.append(torch.log(torch.tensor(mini_batch_size +1 , device='cuda')) - infoNCE_loss(features_a_batch, features_b_batch))
            # mi_estimates.append(infoNCE_loss(features_a_batch, features_b_batch))
        # MI = log(N) - infoNCE
        mi_mean = torch.mean(torch.stack(mi_estimates))
        mi_var = torch.var(torch.stack(mi_estimates))
        return mi_mean, mi_var

    else:
        raise ValueError(f"Unknown method: {method}")



# In[9]:


import os
from entropy_estimators import continuous

def process_models(model_list, val_loader, trigger_loader, val_poisoned_loader, target=None, file_prefix='MI'):
    """
    处理多个模型的函数，并将每个模型的结果保存到文本文件中。

    参数:
    model_list: 一个包含多个模型的列表。
    val_loader: 用于验证的数据加载器。
    trigger_loader: 用于触发的数据加载器。
    val_poisoned_loader: 用于验证的被污染的数据加载器。
    target: 目标类别，可选。
    file_prefix: 保存文件的前缀名。
    """
    poisoned_clean_mi_MINE_means = []
    poisoned_clean_mi_infoNCE_means = []
    poisoned_trigger_mi_MINE_means = []
    filename = os.path.join(path_prefix, f"{file_prefix}.txt")
    
    # 检查文件是否存在，确定从哪个模型开始
    start_index = 0
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        start_index = len(lines)  # 假设每行对应一个模型的输出

    print(f"Starting from model {start_index}...")

    for i in range(start_index, len(model_list)):  # 从上次中断的模型开始

        model = model_list[i]
        # 为每个模型执行特征提取
        clean_features, clean_targets = extract_features(model, val_loader)
        trigger_features, _ = extract_features(model, trigger_loader)
        poisoned_features, poisoned_targets = extract_features(model, val_poisoned_loader)


        # compute the entropy using the k-nearest neighbour approach
        # developed by Kozachenko and Leonenko (1987):
        clean_entropy = continuous.get_h(clean_features, k=8, norm='euclidean', min_dist=1e-5, workers=10)
        poisoned_entropy = continuous.get_h(poisoned_features, k=8, norm='euclidean', min_dist=1e-5, workers=10)
        trigger_entropy = continuous.get_h(trigger_features, k=8, norm='euclidean', min_dist=1e-5, workers=10)
        
        poisoned_clean_mi_infoNCE, _ = get_mi(poisoned_features, clean_features, method='infoNCE')
        # print(f"Model {i} - Poisoned-Clean MI (infoNCE): {poisoned_clean_mi_infoNCE.item()}")
        poisoned_clean_mi_MINE, _ = get_mi(poisoned_features, clean_features, method='MINE')
        poisoned_trigger_mi_MINE, _ = get_mi(poisoned_features, trigger_features, method='MINE')
        # poisoned_clean_mi_MINE, _ = torch.tensor(1, device='cuda'),torch.tensor(1, device='cuda')
        # poisoned_trigger_mi_MINE, _ = torch.tensor(1, device='cuda'),torch.tensor(1, device='cuda')

        poisoned_clean_mi_MINE_means.append(poisoned_clean_mi_MINE.item())
        poisoned_trigger_mi_MINE_means.append(poisoned_trigger_mi_MINE.item())
        poisoned_clean_mi_infoNCE_means.append(poisoned_clean_mi_infoNCE.item())


        # 将结果追加到文本文件
        with open(filename, 'a') as f:
            f.write(f"Model {i} - Poisoned-Clean MI (MINE): {poisoned_clean_mi_MINE.item()} - Poisoned-Trigger MI (MINE): {poisoned_trigger_mi_MINE.item()} - Poisoned-Clean MI (infoNCE): {poisoned_clean_mi_infoNCE.item()} - Clean Entropy: {clean_entropy} - Poisoned Entropy: {poisoned_entropy} - Trigger Entropy: {trigger_entropy}\n")

    return poisoned_clean_mi_MINE_means, poisoned_trigger_mi_MINE_means, poisoned_clean_mi_infoNCE_means

path_prefix = '/workspace/sync/SSL-Backdoor/results/data/0d5Btrainset_trigger14_class_n01820546_mi_entropy'
# 检查路径是否已经存在
if not os.path.exists(path_prefix):
    # 如果不存在，则创建路径
    os.makedirs(path_prefix)

corruptencoder_poisoned_clean_mi_list, corruptencoder_poisoned_trigger_mi_list, corruptencoder_poisoned_clean_mi_list_infoNCE = process_models(corruptencoder_models, val_loader, trigger_loader, val_poisoned_loader, attack_target, file_prefix='MI_corruptencoder')
backog_poisoned_clean_mi_list, backog_poisoned_trigger_mi_list, backog_poisoned_clean_mi_list_infoNCE = process_models(backog_models, val_loader, trigger_loader, val_poisoned_loader, attack_target, file_prefix='MI_backog')
sslbkd_poisoned_clean_mi_list, sslbkd_poisoned_trigger_mi_list, sslbkd_poisoned_clean_mi_list_infoNCE = process_models(sslbkd_models, val_loader, trigger_loader, val_poisoned_loader, attack_target, file_prefix='MI_sslbkd')
init_poisoned_clean_mi_list, init_poisoned_trigger_mi_list, init_poisoned_clean_mi_list_infoNCE = process_models(init_models, val_loader, trigger_loader, val_poisoned_loader, attack_target, file_prefix='MI_init')


# In[ ]:





# In[ ]:


# import matplotlib.pyplot as plt

# def plot_dual_mi_graphs(epochs_list, mi_clean_lists, mi_trigger_lists, labels, title):
#     plt.figure(figsize=(15, 7))

#     # 第一个子图，显示Poisoned-Clean MI
#     plt.subplot(1, 2, 1)
#     for epochs, mi_list, label in zip(epochs_list, mi_clean_lists, labels):
#         plt.plot(epochs, mi_list, marker='o', label=f'{label} Poisoned-Clean')
#     plt.title('Poisoned vs Clean Mutual Information')
#     plt.xlabel('Epochs')
#     plt.ylabel('Mutual Information')
#     plt.grid(True)
#     plt.legend()

#     # 第二个子图，显示Poisoned-Trigger MI
#     plt.subplot(1, 2, 2)
#     for epochs, mi_list, label in zip(epochs_list, mi_trigger_lists, labels):
#         plt.plot(epochs, mi_list, marker='o', label=f'{label} Poisoned-Trigger')
#     plt.title('Poisoned vs Trigger Mutual Information')
#     plt.xlabel('Epochs')
#     plt.ylabel('Mutual Information')
#     plt.grid(True)
#     plt.legend()

#     plt.suptitle(title)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()


# # 组合这些数据
# epochs_list = [sslbkd_epochs, corruptencoder_epochs, backog_epochs]
# mi_clean_lists = [sslbkd_poisoned_clean_mi_list, corruptencoder_poisoned_clean_mi_list, backog_poisoned_clean_mi_list]
# mi_trigger_lists = [sslbkd_poisoned_trigger_mi_list, corruptencoder_poisoned_trigger_mi_list, backog_poisoned_trigger_mi_list]
# labels = ['SSLBKD','corruptencoder','BackOG']

# # 调用绘图函数
# plot_dual_mi_graphs(epochs_list, mi_clean_lists, mi_trigger_lists, labels, 'Mutual Information Analysis for Different Models')

