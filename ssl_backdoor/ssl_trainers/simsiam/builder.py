# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl_backdoor.utils.model_utils import transform_encoder_for_small_dataset, remove_task_head_for_encoder
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
        encoder_name = str(base_encoder).lower()
        self.encoder = base_encoder(num_classes=dim)
        
        if "squeezenet" in encoder_name:
            # 对于squeezenet，跳过特殊处理，直接使用dim作为channel_dim
            channel_dim = dim
        else:
            # 对于其他网络，按原来的方式处理
            channel_dim = SimSiam.get_channel_dim(self.encoder)
            self.encoder = transform_encoder_for_small_dataset(self.encoder, dataset)
            self.encoder = remove_task_head_for_encoder(self.encoder)

        self.projector = SimSiamProjectionHead(input_dim=channel_dim, hidden_dim=channel_dim, output_dim=dim)
        self.predictor = SimSiamPredictionHead(input_dim=dim, hidden_dim=pred_dim, output_dim=dim)


    @staticmethod
    def get_channel_dim(encoder: nn.Module) -> int:
        def get_channel_dim_from_sequential(sequential: nn.Sequential) -> int:
            for module in sequential:
                if isinstance(module, nn.Linear):
                    return module.in_features
                elif isinstance(module, nn.Conv2d):
                    return module.in_channels
            raise ValueError("没有在Sequential中找到Linear层或Conv2d层")
            
        if hasattr(encoder, 'fc'):
            return encoder.fc.weight.shape[1]
        elif hasattr(encoder, 'head'):
            return encoder.head.weight.shape[1]
        elif hasattr(encoder, 'heads'):
            # 处理Vision Transformer中的heads属性
            if hasattr(encoder.heads, 'head'):
                return encoder.heads.head.in_features
            # 遍历Sequential中的层
            if isinstance(encoder.heads, nn.Sequential):
                for name, module in encoder.heads.named_children():
                    if name == 'head' or name == 'pre_logits':
                        return module.in_features
                # 如果没有找到head或pre_logits，尝试获取第一个Linear层
                return get_channel_dim_from_sequential(encoder.heads)
        elif hasattr(encoder, 'classifier'):
            if isinstance(encoder.classifier, nn.Sequential):
                return get_channel_dim_from_sequential(encoder.classifier)
            else:
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
