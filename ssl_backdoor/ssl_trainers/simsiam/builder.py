# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl_backdoor.ssl_trainers.utils import transform_encoder_for_small_dataset, remove_task_head_for_encoder
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, dataset=None):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        channel_dim = SimSiam.get_channel_dim(self.encoder)
        self.encoder = transform_encoder_for_small_dataset(self.encoder, dataset)
        self.encoder = remove_task_head_for_encoder(self.encoder)


        self.projector = SimSiamProjectionHead(input_dim=channel_dim, hidden_dim=channel_dim, output_dim=dim)
        self.predictor = SimSiamPredictionHead(input_dim=dim, hidden_dim=pred_dim, output_dim=dim)


    @staticmethod
    def get_channel_dim(encoder: nn.Module) -> int:
        if hasattr(encoder, 'fc'):
            return encoder.fc.weight.shape[1]
        elif hasattr(encoder, 'head'):
            return encoder.head.weight.shape[1]
        elif hasattr(encoder, 'classifier'):
            return encoder.classifier.weight.shape[1]
        else:
            raise NotImplementedError('MLP projection head not found in encoder')
        

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        z1 = self.projector(z1)
        z2 = self.projector(z2)

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        z1, z2 = z1.detach(), z2.detach()

        loss = -(F.cosine_similarity(p1, z2).mean() + F.cosine_similarity(p2, z1).mean()) * 0.5

        return loss
