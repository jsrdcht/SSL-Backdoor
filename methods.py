import copy
import math

import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead, SimSiamPredictionHead, SimSiamProjectionHead, MoCoProjectionHead
from lightly.models.modules.heads import SimCLRProjectionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from lightly.loss import NegativeCosineSimilarity, NTXentLoss

class ExtendedBYOLProjectionHead(BYOLProjectionHead):
    """Extended BYOL Projection Head with an additional layer."""

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 256
    ):
        super(BYOLProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),  # 额外添加的一层
                (hidden_dim, output_dim, None, None),
            ]
        )

class MoCo(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        model = models.__dict__[args.arch]()
        fea_dim = model.fc.in_features
        model.fc = nn.Identity()

        self.backbone = model
        self.projection_head = MoCoProjectionHead(fea_dim, fea_dim, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(temperature=0.2, memory_bank_size=(65536, 128))

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def training_step(self, batch, batch_idx):
        # momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        momentum = 0.999
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        x_query, x_key = batch[0]
        query = self.forward(x_query)
        key = self.forward_momentum(x_key)
        loss = self.criterion(query, key)

        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.06, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        return [optimizer], [scheduler]
    
class BYOL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        model = models.__dict__[args.arch]()
        if 'imagenet' not in args.dataset:
            print("Using custom conv1 for small datasets")
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if args.dataset == "cifar10" or args.dataset == "cifar100":
            print("Using custom maxpool for cifar datasets")
            model.maxpool = nn.Identity()
        model.fc = nn.Identity()

        self.backbone = model
        self.projection_head = ExtendedBYOLProjectionHead(512, 1024, 128)
        self.prediction_head = BYOLPredictionHead(128, 1024, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # update_momentum(self.backbone, self.backbone_momentum, m=0)
        # update_momentum(self.projection_head, self.projection_head_momentum, m=0)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, self.args.epochs, 0.99, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        (x0, x1) = batch[0]
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))

        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True)

        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']  # 假设所有组使用相同的学习率
        self.log("learning_rate", lr, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=1e-6)

    #     # 余弦退火调度器
    #     cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, 
    #         T_max=self.args.epochs  # 调度周期为总训练的 epoch 数
    #     )

    #     # 返回优化器和调度器
    #     return [optimizer], [
    #         {
    #             'scheduler': cosine_scheduler, 
    #             'interval': 'epoch',  # 每个 epoch 更新一次
    #             'frequency': 1,
    #             'name': 'cosine_scheduler'
    #         }
    #     ]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=1e-6)
        
        # 线性预热调度器(前500步)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1e-8, 
            end_factor=1.0, 
            total_iters=500
        )

        # 余弦退火调度器（预热完成后使用）
        # 假设这里T_max以epoch为单位来退火，你可以根据需求改为step级别。
        # 如果你以epoch为间隔进行scheduler step操作，T_max可以是self.args.epochs。
        # 如果以step为间隔，你需要知道总steps，再进行设置。
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.args.epochs,  # 依据实际训练长度修改
            eta_min=0
        )

        # 使用SequentialLR将两个scheduler串联起来。
        # 当训练step数 < 500时只执行warmup_scheduler，
        # 达到500后自动切换到cosine_scheduler。
        combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[500]  # 在500 step后进入第二个scheduler
        )

        # 因为warmup需要按step来进行，所以把interval设为step
        return [optimizer], [{
            'scheduler': combined_scheduler,
            'interval': 'step',   # 每个step都对scheduler进行step
            'frequency': 1,
            'name': 'combined_scheduler'
        }]
    
class SimSiam(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        model = models.__dict__[args.arch](zero_init_residual=True)
        fea_dim = model.fc.in_features
        model.fc = nn.Identity()

        self.backbone = model

        proj_hidden_dim = fea_dim
        pred_input_dim = 2048
        pred_hidden_dim = 512
        pred_output_dim = 2048
        
        self.projection_head = SimSiamProjectionHead(fea_dim, proj_hidden_dim, pred_input_dim)
        self.prediction_head = SimSiamPredictionHead(pred_input_dim, pred_hidden_dim, pred_output_dim)
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1) = batch[0]
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))

        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True)
        # Log the current learning rate
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']  # 假设所有组使用相同的学习率
        self.log("learning_rate", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        epochs = self.args.epochs
        init_lr = 0.05
    
        optim_params = [
            {'params': self.backbone.parameters(), 'fix_lr': False},
            {'params': self.projection_head.parameters(), 'fix_lr': False},
            {'params': self.prediction_head.parameters(), 'fix_lr': True}
        ]
        
        optimizer = torch.optim.SGD(optim_params, lr=init_lr, momentum=0.9, weight_decay=1e-4)
    
        # 定义余弦退火调度函数
        def cosine_annealing_schedule(epoch):
            return 0.5 * (1 + math.cos(math.pi * epoch / epochs))
    
        # 为每个参数组定义对应的 lr_lambda 函数
        lr_lambdas = []
        for param_group in optimizer.param_groups:
            if param_group.get('fix_lr', False):
                lr_lambdas.append(lambda epoch: 1.0)  # 学习率保持不变
            else:
                lr_lambdas.append(cosine_annealing_schedule)
    
        # 创建学习率调度器
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
    
        return [optimizer], [scheduler]
    
class SimCLR(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args

        model = models.__dict__[args.arch]()
        if 'imagenet' not in args.dataset:
            print("Using custom conv1 for small datasets")
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if args.dataset == "cifar10" or args.dataset == "cifar100":
            print("Using custom maxpool for cifar datasets")
            model.maxpool = nn.Identity()
        hidden_dim = model.fc.in_features

        # resnet = models.__dict__[args.arch]()
        # hidden_dim = resnet.fc.in_features

        # self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone = model
        self.backbone.fc = nn.Identity()

        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)

        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        weight_decay = getattr(self.args, 'weight_decay', 1e-4)

        optim = torch.optim.SGD(
            self.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=weight_decay
        )
        # Warmup scheduler for first 10 epochs
        def warmup_scheduler(epoch):
            if epoch < 10:
                return epoch / 10  # Gradually increase LR
            else:
                return 1.0  # Use normal LR after warmup

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_scheduler)

        # Cosine annealing scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.args.epochs - 10)

        # Combine warmup and cosine annealing
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optim,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[10]
        )
        return [optim], [scheduler]