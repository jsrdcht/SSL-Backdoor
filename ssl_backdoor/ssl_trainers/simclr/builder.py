import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl_backdoor.utils.model_utils import transform_encoder_for_small_dataset, remove_task_head_for_encoder
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss


class SimCLR(nn.Module):
    """
        
    """
    def __init__(self, base_encoder, dim=512, proj_dim=128, dataset=None):
        """
            
            
        """
        super(SimCLR, self).__init__()
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        channel_dim = SimCLR.get_channel_dim(self.encoder)
        self.encoder = transform_encoder_for_small_dataset(self.encoder, dataset)
        self.encoder = remove_task_head_for_encoder(self.encoder)
        self.projector = SimCLRProjectionHead(input_dim=channel_dim, hidden_dim=dim, output_dim=proj_dim)

        self.criterion = NTXentLoss()

    @staticmethod
    def get_channel_dim(encoder: nn.Module) -> int:
        if hasattr(encoder, 'fc'):
            return encoder.fc.weight.shape[1]
        elif hasattr(encoder, 'head'):
            return encoder.head.weight.shape[1]
        elif hasattr(encoder, 'classifier'):
            return encoder.classifier.weight.shape[1]
        else:
            raise NotImplementedError('MLP projection head was not found in the encoder')

    def forward(self, x1, x2):
        """
            
            
            
            
            
        """
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        loss = self.criterion(z1, z2)

        return loss