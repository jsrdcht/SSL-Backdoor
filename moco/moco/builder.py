import typing
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_last_linear_layer(module: nn.Module) -> typing.Optional[nn.Linear]:
    """Recursively find the last nn.Linear layer in a Sequential module."""
    if isinstance(module, nn.Sequential):
        for layer in reversed(module):
            if isinstance(layer, nn.Linear):
                return layer
            elif isinstance(layer, nn.Sequential):
                return find_last_linear_layer(layer)
    elif isinstance(module, nn.Linear):
        return module
    return None

class MoCo(nn.Module):
    def __init__(self, base_encoder: nn.Module, dim: int = 128, K: int = 65536, m: float = 0.999,
                 contr_tau: float = 0.07, align_alpha: typing.Optional[int] = None, 
                 unif_t: typing.Optional[float] = None, unif_intra_batch: bool = True):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.contr_tau = contr_tau
        self.align_alpha = align_alpha
        self.unif_t = unif_t
        self.unif_intra_batch = unif_intra_batch

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        self._initialize_encoders()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if contr_tau is not None:
            self.register_buffer('scalar_label', torch.zeros((), dtype=torch.long))
        else:
            self.register_parameter('scalar_label', None)

    def _initialize_encoders(self):
        if hasattr(self.encoder_q, 'fc'):
            self._add_mlp_projection_head(self.encoder_q, self.encoder_k, 'fc')
        elif hasattr(self.encoder_q, 'head'):
            self._add_mlp_projection_head(self.encoder_q, self.encoder_k, 'head')
        elif hasattr(self.encoder_q, 'classifier'):
            self._add_mlp_projection_head(self.encoder_q, self.encoder_k, 'classifier')
        else:
            raise NotImplementedError('MLP projection head not found in encoder')

    def _add_mlp_projection_head(self, encoder_q, encoder_k, attr):
        dim_mlp = getattr(encoder_q, attr).weight.shape[1]
        setattr(encoder_q, attr, nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), getattr(encoder_q, attr)))
        setattr(encoder_k, attr, nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), getattr(encoder_k, attr)))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
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
        q = self.encoder_q(im_q)
        return F.normalize(q, dim=1)

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor):
        q = self.forward_features(im_q)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)
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