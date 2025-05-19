import torch.nn.functional as F
import random
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def extract_features(model, loader, class_index=None):
    """
    Extracts features from the model using the given loader and saves them to a file.

    Args:
    model (torch.nn.Module): The model from which to extract features.
    loader (torch.utils.data.DataLoader): The DataLoader for input data.
    class_index (int): The index of the class to extract features for. If None, all classes are used.
    """
    model.eval()
    device = next(model.parameters()).device

    features = []
    target_list = []
    

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(loader)):
            if class_index is not None:
                mask = targets == class_index
                inputs = inputs[mask]
                targets = targets[mask]

            inputs = inputs.to(device)
            output = model(inputs)
            output = F.normalize(output, dim=1)
            features.append(output.detach().cpu())
            target_list.append(targets)
    
    features = torch.cat(features, dim=0)
    targets = torch.cat(target_list, dim=0)

    
    return features, targets

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def get_channels(arch):
    if arch == 'alexnet':
        c = 4096
    elif arch == 'pt_alexnet':
        c = 4096
    elif arch == 'resnet50':
        c = 2048
    elif 'resnet18' in arch:
        c = 512
    elif 'densenet121' in arch:
        c = 1024
    elif arch == 'mobilenet':
        c = 1280
    elif arch == 'MobileNetV2':
        c = 1280
    elif arch == 'SqueezeNet':
        c = 2048
    elif arch == 'resnet50x5_swav':
        c = 10240
    elif arch == 'vit_b_16':
        c = 768
    elif arch == 'swin_s':
        c = 768
    else:
        raise ValueError('arch not found: ' + arch)
    return c

def knn_evaluate(model, train_loader, test_loader, device):
    model.eval()
    model.to(device)
    feature_bank = []
    labels = []
    
    # 构建特征库和标签
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            feature = model(data).flatten(start_dim=1)
            feature_bank.append(feature.cpu())
            labels.append(target.cpu())
    
    feature_bank = torch.cat(feature_bank, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()  # 转换为 NumPy 数组
    
    # 训练 KNN
    knn = NearestNeighbors(n_neighbors=200, metric='cosine')
    knn.fit(feature_bank)
    
    total_correct = 0
    total_num = 0
    all_preds = []
    all_targets_list = []

    # 评估阶段``
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            feature = model(data).flatten(start_dim=1)
            feature = feature.cpu().numpy()
            
            distances, indices = knn.kneighbors(feature)
            
            # 使用 NumPy 进行索引
            retrieved_neighbors = labels[indices]  # shape: [batch_size, n_neighbors]
            
            # 计算预测标签（使用众数）
            pred_labels = np.squeeze(np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, retrieved_neighbors))
            
            # 将预测标签转换为 PyTorch 张量
            pred_labels = torch.tensor(pred_labels, device='cpu')  # 使用 CPU 进行比较
            
            # 计算正确预测数量
            total_correct += (pred_labels == target.cpu()).sum().item()
            total_num += data.size(0)
            all_preds.append(pred_labels)
            all_targets_list.append(target)
    
    accuracy = total_correct / total_num
    print(f"[knn_evaluate] Total accuracy: {accuracy * 100:.2f}%")
    all_preds = torch.cat(all_preds, dim=0)
    all_targets_list = torch.cat(all_targets_list, dim=0)
    return accuracy, all_preds, all_targets_list


def extract_config_by_prefix(config_dict, prefix):
    """
    从配置字典中提取特定前缀的键值对
    
    Args:
        config_dict (dict): 配置字典
        prefix (str): 键前缀
        
    Returns:
        dict: 包含所有以指定前缀开头的键值对的字典
    """
    # 如果前缀恰好是字典中的一个键，并且对应值是字典，则直接返回该子字典
    if prefix in config_dict and isinstance(config_dict[prefix], dict):
        return config_dict[prefix]
    
    # 否则，查找所有以该前缀开头的键
    result = {}
    for key, value in config_dict.items():
        if key.startswith(f"{prefix}"):
            key = key[len(prefix):]
            result[key] = value
    
    return result


def set_seed(seed):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True