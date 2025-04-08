import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead

from ssl_backdoor.ssl_trainers.utils import transform_encoder_for_small_dataset, remove_task_head_for_encoder


class BYOL(nn.Module):
    """
    Build a BYOL model.
    """
    def __init__(self, base_encoder, dim=2048, proj_dim=1024, pred_dim=128, tau=0.99, dataset=None):
        """
        dim: feature dimension (default: 2048)
        proj_dim: hidden dimension of the projector (default: 1024)
        pred_dim: hidden dimension of the predictor (default: 128)
        tau: target network momentum (default: 0.99)
        """
        super(BYOL, self).__init__()

        # create the online encoder
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder = transform_encoder_for_small_dataset(self.encoder, dataset)
        self.encoder = remove_task_head_for_encoder(self.encoder)

        # create the target encoder
        self.momentum_encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.momentum_encoder = transform_encoder_for_small_dataset(self.momentum_encoder, dataset)
        self.momentum_encoder = remove_task_head_for_encoder(self.momentum_encoder)

        
        self.projector = BYOLProjectionHead(input_dim=dim, hidden_dim=proj_dim, output_dim=pred_dim)
        self.momentum_projector = BYOLProjectionHead(input_dim=dim, hidden_dim=proj_dim, output_dim=pred_dim)
        self.predictor = BYOLPredictionHead(input_dim=pred_dim, hidden_dim=proj_dim, output_dim=pred_dim)

        # disable gradient for target encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        for param in self.momentum_projector.parameters():
            param.requires_grad = False
            
        # momentum coefficient
        self.tau = tau
        
        # copy parameters from online to target encoder
        self.update_target(0)

    def update_target(self, progress=None):
        """
        Update target network parameters using momentum update rule
        If progress is provided, use cosine schedule
        """
        tau = self.tau
        if progress is not None:
            # cosine schedule as in the original BYOL implementation
            tau = 1 - (1 - self.tau) * (math.cos(math.pi * progress) + 1) / 2
            
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * tau + param_q.data * (1. - tau)
        for param_q, param_k in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            param_k.data = param_k.data * tau + param_q.data * (1. - tau)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            loss: BYOL loss
        """
        # compute online features
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        z1_proj = self.projector(z1)
        z2_proj = self.projector(z2)

        # compute online predictions
        p1 = self.predictor(z1_proj) # NxC
        p2 = self.predictor(z2_proj) # NxC
        # compute target features (no gradient)
        with torch.no_grad():
            z1_target = self.momentum_encoder(x1)
            z2_target = self.momentum_encoder(x2)

            z1_target_proj = self.momentum_projector(z1_target)
            z2_target_proj = self.momentum_projector(z2_target)

        # normalize for cosine similarity
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        z1_target_proj = F.normalize(z1_target_proj, dim=-1)
        z2_target_proj = F.normalize(z2_target_proj, dim=-1)

        # BYOL loss
        loss = 4 - 2 * (p1 * z2_target_proj).sum(dim=-1).mean() - 2 * (p2 * z1_target_proj).sum(dim=-1).mean()

        return loss 