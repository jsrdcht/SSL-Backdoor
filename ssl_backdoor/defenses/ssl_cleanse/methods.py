import torch
import torch.nn.functional as F

def neg_cosine_similarity(x, y):
    """
    Compute the negative cosine similarity loss.
    
    Args:
        x: First feature vector.
        y: Second feature vector.
    
    Returns:
        Tensor containing the negative cosine similarity loss.
    """
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return -torch.mean(torch.sum(x * y, dim=1)) 
