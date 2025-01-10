""" DINOv2 model adapter
"""
import beartype
from jaxtyping import jaxtyped
from collections import OrderedDict

import torch
import torch.nn as nn

from transformers import AutoModel


@jaxtyped(typechecker=beartype.beartype)
class DINOv2Model(nn.Module):
    """
    Add adapter head to DINOv2.
    """
    def __init__(
        self,
        model_name: str,
        embed_dim: int,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.embed_dim = embed_dim

        prev_chs = self.backbone.config.hidden_size
        self.backbone.embeddings.mask_token.requires_grad_(False)

        if embed_dim > 0:
            head_layers = OrderedDict()
            head_layers['drop'] = nn.Dropout(0.)
            head_layers['proj'] = nn.Linear(prev_chs, embed_dim, bias=False)
            self.head = nn.Sequential(head_layers)
        else:
            self.head = nn.Identity()

    def get_cast_dtype(self) -> torch.dtype:
        return self.head.proj.weight.dtype

    def forward(self, x):
        _, x = self.backbone(x, return_dict=False)
        if self.head is not None:
            x = self.head(x)
        return x
