import os

import torch
import torch.nn as nn
from einops import rearrange
from torch.profiler import record_function

from osu_fusion.modules.attention import Attention
from osu_fusion.modules.utils import dummy_context_manager

DEBUG = os.environ.get("DEBUG", False)


class FeedForward(nn.Sequential):
    def __init__(self: "FeedForward", dim: int, dim_mult: int = 2) -> None:
        inner_dim = dim * dim_mult
        super().__init__(
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, dim),
        )

    def forward(self: "FeedForward", x: torch.Tensor) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("FeedForward")
        with context_manager:
            return super().forward(x)


class TransformerBlock(nn.Module):
    def __init__(
        self: "TransformerBlock",
        dim: int,
        ff_mult: int = 2,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.attn = Attention(
            dim,
            attn_dim_head,
            attn_heads,
            attn_kv_heads,
            attn_context_len,
        )
        self.ff = FeedForward(dim, ff_mult)

    def forward(self: "TransformerBlock", x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b d n -> b n d")
        x = self.attn(x) + x
        x = self.ff(x) + x
        return rearrange(x, "b n d -> b d n")
