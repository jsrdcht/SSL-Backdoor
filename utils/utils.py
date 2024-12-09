import torch.nn.functional as F
import random
import torch

from PIL import Image
from tqdm import tqdm

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
    elif arch == 'mobilenet':
        c = 1280
    elif arch == 'resnet50x5_swav':
        c = 10240
    elif arch == 'vit_base_patch16':
        c = 768
    elif arch == 'swin_s':
        c = 768
    else:
        raise ValueError('arch not found: ' + arch)
    return c