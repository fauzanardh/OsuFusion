from typing import Dict, List, Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat

from osu_fusion.modules.attention import MultiHeadAttention
from osu_fusion.modules.causal_convolution import CausalConv1d


class Always:
    def __init__(self: "Always", value: int) -> None:
        self.value = int

    def __call__(self: "Always", *args: List, **kwargs: Dict) -> int:
        return self.value


class SqueezeExcite(nn.Module):
    def __init__(self: "SqueezeExcite", dim: int, reduction_factor: int = 4, dim_minimum: int = 8) -> None:
        super().__init__()
        hidden_dim = max(dim // reduction_factor, dim_minimum)
        self.layers = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self: "SqueezeExcite", x: torch.Tensor) -> torch.Tensor:
        seq, device = x.shape[-2], x.device

        cum_sum = x.cumsum(dim=-2)
        denominator = torch.arange(1, seq + 1, device=device).float()
        mean = cum_sum / rearrange(denominator, "n -> n 1")

        return x * self.layers(mean)


class GlobalContext(nn.Module):
    """Attention-esque squeeze-excite module"""

    def __init__(self: "GlobalContext", dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.to_k = CausalConv1d(dim_in, 1, 1)
        inner_dim = max(3, dim_out // 2)

        self.layers = nn.Sequential(
            CausalConv1d(dim_in, inner_dim, 1),
            nn.SiLU(),
            CausalConv1d(inner_dim, dim_out, 1),
            nn.Sigmoid(),
        )

    def forward(self: "GlobalContext", x: torch.Tensor) -> torch.Tensor:
        context = self.to_k(x)
        out = torch.einsum("b i d, b j d -> b i j", x, context.softmax(dim=-1))
        return self.layers(out)


class ResidualBlock(nn.Module):
    def __init__(
        self: "ResidualBlock",
        dim_in: int,
        dim_out: int,
        dim_emb: int,
        dilation: int,
        kernel_size: int = 7,
        squeeze_excite: bool = True,
    ) -> None:
        super().__init__()

        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_emb, dim_in),
        )
        self.layers = nn.Sequential(
            CausalConv1d(dim_in, dim_out, kernel_size, dilation=dilation),
            nn.ReLU(),
            CausalConv1d(dim_out, dim_out, 1),
            nn.ReLU(),
            SqueezeExcite(dim_out) if squeeze_excite else nn.Identity(),
        )
        self.res_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self: "ResidualBlock", x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = x

        time_emb = self.emb(time_emb).unsqueeze(-1)
        h = self.layers(h + time_emb)

        return h + self.res_conv(x)


class Block(nn.Module):
    def __init__(
        self: "Block",
        dim_in: int,
        dim_out: int,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim_in) if norm else nn.Identity()
        self.activation = nn.SiLU()
        self.res_conv = CausalConv1d(dim_in, dim_out, 7)

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
            )
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out, dim_out)

        self.gca = GlobalContext(dim_in, dim_out) if use_gca else Always(1)
        self.res_conv = CausalConv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(
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
            context = repeat(context, "b d -> b n d", n=h.shape[-1])
            h = self.cross_attention(h, context) + h
            h = rearrange(h, "b n d -> b d n")

        h = self.block2(h, scale_shift=scale_shift)
        h = h * self.gca(x)

        return h + self.res_conv(x)
