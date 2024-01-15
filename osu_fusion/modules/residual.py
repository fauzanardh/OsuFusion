from typing import Dict, List, Optional

import torch
import torch.nn as nn
from einops import rearrange

from osu_fusion.modules.attention import MultiHeadAttention


class Always:
    def __init__(self: "Always", value: int) -> None:
        self.value = value

    def __call__(self: "Always", *args: List, **kwargs: Dict) -> int:
        return self.value


class GlobalContext(nn.Module):
    """Attention-esque squeeze-excite module"""

    def __init__(self: "GlobalContext", dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.to_k = nn.Conv1d(dim_in, 1, 1)
        inner_dim = max(3, dim_out // 2)

        self.layers = nn.Sequential(
            nn.Conv1d(dim_in, inner_dim, 1),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim_out, 1),
            nn.Sigmoid(),
        )

    def forward(self: "GlobalContext", x: torch.Tensor) -> torch.Tensor:
        context = self.to_k(x)
        out = torch.einsum("b i d, b j d -> b i j", x, context.softmax(dim=-1))
        return self.layers(out)


class Block(nn.Module):
    def __init__(
        self: "Block",
        dim_in: int,
        dim_out: int,
        norm: bool = True,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(norm_groups, dim_in) if norm else nn.Identity()
        self.activation = nn.SiLU()
        self.res_conv = nn.Conv1d(dim_in, dim_out, 3, padding=1)

    def forward(self: "Block", x: torch.Tensor, scale_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.res_conv(x)


class ResidualBlockV2(nn.Module):
    def __init__(
        self: "ResidualBlockV2",
        dim_in: int,
        dim_out: int,
        dim_emb: int,
        dim_context: Optional[int] = None,
        attn_dim_head: int = 32,
        attn_heads: int = 8,
        attn_dropout: float = 0.1,
        attn_sdpa: bool = True,
        attn_use_rotary_emb: bool = True,
        use_gca: bool = True,
    ) -> None:
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_emb, dim_out * 2),
        )
        if dim_context is not None:
            self.cross_attention = MultiHeadAttention(
                dim_out,
                dim_context=dim_context,
                dim_head=attn_dim_head,
                heads=attn_heads,
                dropout=attn_dropout,
                sdpa=attn_sdpa,
                is_cross_attention=True,
                use_rotary_emb=attn_use_rotary_emb,
            )
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out, dim_out)

        self.gca = GlobalContext(dim_in, dim_out) if use_gca else Always(1)
        self.res_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

        self.gradient_checkpointing = False

    def forward_body(
        self: "ResidualBlockV2",
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        time_emb = self.time_mlp(time_emb)
        time_emb = rearrange(time_emb, "b n -> b n 1")
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)

        if hasattr(self, "cross_attention") and context is not None:
            h = rearrange(h, "b d n -> b n d")
            h = self.cross_attention(h, context) + h
            h = rearrange(h, "b n d -> b d n")

        h = self.block2(h, scale_shift=scale_shift)
        h = h * self.gca(x)

        return h + self.res_conv(x)

    def forward(
        self: "ResidualBlockV2",
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, time_emb, context, use_reentrant=True)
        else:
            return self.forward_body(x, time_emb, context)
