import os 
import random
import numpy as np
import torch
from PIL import Image
import torchvision
import time
import sys





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def set_seed(seed):
    # for reproducibility. 
    # note that pytorch is not completely reproducible 
    # https://pytorch.org/docs/stable/notes/randomness.html  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed() # dataloader multi processing 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

def save_model(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_model(model, path):
    model_path = os.path.join(path, 'epoch_301.pth.tar')
    checkpoint = torch.load(model_path, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    
    new_state_dict = {}
    for k in list(state_dict.keys()):
        if 'backbone' in k and 'distill' not in k:
            if k.startswith('module.'):
                new_state_dict[k[len("module.")+ len("backbone."):]] = state_dict[k]
            else:
                if 'backbone_k' not in k:
                    new_state_dict[k[len("backbone."):]] = state_dict[k]

    model.load_state_dict(new_state_dict, strict=True)
    print("=> loaded pre-trained model '{}'".format(model_path))
    print(checkpoint['epoch'])
    
    return model