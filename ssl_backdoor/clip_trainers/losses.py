"""Standard CLIP bidirectional InfoNCE loss, supports DDP global negatives."""
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F


def _gather_with_grad(tensor):
    """all_gather features from all ranks, gradients fully back-propagated via collective communication (open_clip gather_with_grad)."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    return torch.cat(torch.distributed.nn.all_gather(tensor.contiguous()), dim=0)


class ClipLoss(nn.Module):
    """Symmetric InfoNCE in both image->text and text->image directions, computed on global similarity matrix."""

    def forward(self, image_embeds, text_embeds, logit_scale):
        all_images = _gather_with_grad(F.normalize(image_embeds, dim=-1))
        all_texts = _gather_with_grad(F.normalize(text_embeds, dim=-1))

        logits = logit_scale.exp() * all_images @ all_texts.t()
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.t(), labels)) / 2
