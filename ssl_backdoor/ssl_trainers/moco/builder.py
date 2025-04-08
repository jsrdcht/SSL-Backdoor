import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightly.models.modules.heads import MoCoProjectionHead
from ssl_backdoor.ssl_trainers.utils import remove_task_head_for_encoder, transform_encoder_for_small_dataset

# def find_last_linear_layer(module: nn.Module) -> typing.Optional[nn.Linear]:
#     """Recursively find the last nn.Linear layer in a Sequential module."""
#     if isinstance(module, nn.Sequential):
#         for layer in reversed(module):
#             if isinstance(layer, nn.Linear):
#                 return layer
#             elif isinstance(layer, nn.Sequential):
#                 return find_last_linear_layer(layer)
#     elif isinstance(module, nn.Linear):
#         return module
#     return None

class MoCo(nn.Module):
    def __init__(self, base_encoder: nn.Module, dim: int = 128, K: int = 65536, m: float = 0.999,
                 contr_tau: float = 0.07, align_alpha: typing.Optional[int] = None, 
                 unif_t: typing.Optional[float] = None, unif_intra_batch: bool = True, dataset: str = None):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.contr_tau = contr_tau
        self.align_alpha = align_alpha
        self.unif_t = unif_t
        self.unif_intra_batch = unif_intra_batch

        self.encoder = base_encoder(num_classes=dim)
        channel_dim = MoCo.get_channel_dim(self.encoder)
        self.moumentum_encoder = base_encoder(num_classes=dim)
        self.encoder = transform_encoder_for_small_dataset(self.encoder, dataset)
        self.moumentum_encoder = transform_encoder_for_small_dataset(self.moumentum_encoder, dataset)
        self.encoder = remove_task_head_for_encoder(self.encoder)
        self.moumentum_encoder = remove_task_head_for_encoder(self.moumentum_encoder)

        # create projector head
        self.projector = MoCoProjectionHead(input_dim=channel_dim, hidden_dim=channel_dim, output_dim=dim)
        self.momentum_projector = MoCoProjectionHead(input_dim=channel_dim, hidden_dim=channel_dim, output_dim=dim)

        for param_q, param_k in zip(self.encoder.parameters(), self.moumentum_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if contr_tau is not None:
            self.register_buffer('scalar_label', torch.zeros((), dtype=torch.long))
        else:
            self.register_parameter('scalar_label', None)

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

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.moumentum_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        for param_q, param_k in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this

        idx_shuffle = torch.randperm(batch_size_all).to(x.device)
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)

        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x: torch.Tensor, idx_unshuffle: torch.Tensor) -> torch.Tensor:
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this

        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_features(self, im_q: torch.Tensor) -> torch.Tensor:
        features = self.encoder(im_q)
        q = self.projector(features)
        return F.normalize(q, dim=1)

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor):
        q = self.forward_features(im_q)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k_features = self.moumentum_encoder(im_k)
            k = self.momentum_projector(k_features)
            k = F.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        moco_loss_ctor_dict = {}

        def get_q_bdot_k():
            if not hasattr(get_q_bdot_k, 'result'):
                get_q_bdot_k.result = (q * k).sum(dim=1)
            return get_q_bdot_k.result

        def get_q_dot_queue():
            if not hasattr(get_q_dot_queue, 'result'):
                get_q_dot_queue.result = q @ self.queue.clone().detach()
            return get_q_dot_queue.result

        if self.contr_tau is not None:
            l_pos = get_q_bdot_k().unsqueeze(-1)
            l_neg = get_q_dot_queue()
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.contr_tau

            moco_loss_ctor_dict['logits_contr'] = logits
            moco_loss_ctor_dict['loss_contr'] = F.cross_entropy(logits, self.scalar_label.expand(logits.shape[0]))

        if self.align_alpha is not None:
            if self.align_alpha == 2:
                moco_loss_ctor_dict['loss_align'] = 2 - 2 * get_q_bdot_k().mean()
            elif self.align_alpha == 1:
                moco_loss_ctor_dict['loss_align'] = (q - k).norm(dim=1, p=2).mean()
            else:
                moco_loss_ctor_dict['loss_align'] = (2 - 2 * get_q_bdot_k()).pow(self.align_alpha / 2).mean()

        if self.unif_t is not None:
            sq_dists = (2 - 2 * get_q_dot_queue()).flatten()
            if self.unif_intra_batch:
                sq_dists = torch.cat([sq_dists, torch.pdist(q, p=2).pow(2)])
            moco_loss_ctor_dict['loss_unif'] = sq_dists.mul(-self.unif_t).exp().mean().log()

        self._dequeue_and_enqueue(k)

        return moco_loss_ctor_dict['loss_contr']

@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)