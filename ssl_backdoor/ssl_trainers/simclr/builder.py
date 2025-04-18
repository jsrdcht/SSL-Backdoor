import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl_backdoor.utils.model_utils import transform_encoder_for_small_dataset, remove_task_head_for_encoder
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss


class SimCLR(nn.Module):
    """
    构建一个SimCLR模型。
    """
    def __init__(self, base_encoder, dim=512, proj_dim=128, dataset=None):
        """
        dim: 特征维度 (默认: 2048)
        proj_dim: 投影头输出维度 (默认: 128)
        """
        super(SimCLR, self).__init__()

        # 创建编码器
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        channel_dim = SimCLR.get_channel_dim(self.encoder)
        self.encoder = transform_encoder_for_small_dataset(self.encoder, dataset)
        self.encoder = remove_task_head_for_encoder(self.encoder)

        # 创建投影头 (MLP)
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
            raise NotImplementedError('MLP投影头在编码器中未找到')

    def forward(self, x1, x2):
        """
        输入:
            x1: 图像第一视角
            x2: 图像第二视角
        输出:
            对比损失
        """
        # 计算两个视角的特征
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        # 通过投影头
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # 归一化表示以进行余弦相似度
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        loss = self.criterion(z1, z2)

        return loss