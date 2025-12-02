import torch
import torch.nn.functional as F

def neg_cosine_similarity(x, y):
    """
    计算负余弦相似度损失
    
    参数:
        x: 第一个特征向量
        y: 第二个特征向量
    
    返回:
        tensor: 负余弦相似度损失
    """
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return -torch.mean(torch.sum(x * y, dim=1)) 