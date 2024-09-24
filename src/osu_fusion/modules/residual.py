import os
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.profiler import record_function

from osu_fusion.modules.utils import dummy_context_manager

DEBUG = os.environ.get("DEBUG", False)


class GlobalContext(nn.Module):
    """Attention-esque squeeze-excite module"""

    def __init__(self: "GlobalContext", dim_in: int, dim_out: int, reduction: int = 2, dim_min: int = 8) -> None:
        super().__init__()
        self.to_k = nn.Conv1d(dim_in, 1, 1)
        inner_dim = max(dim_min, dim_out // reduction)

        self.layers = nn.Sequential(
            nn.Conv1d(dim_in, inner_dim, 1),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim_out, 1),
            nn.Sigmoid(),
        )

    def forward_body(self: "GlobalContext", x: torch.Tensor) -> torch.Tensor:
        context = self.to_k(x)
        out = torch.einsum("b i d, b j d -> b i j", x, context.softmax(dim=-1))
        return self.layers(out)

    def forward(self: "GlobalContext", x: torch.Tensor) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("GlobalContext")
        with context_manager:
            return self.forward_body(x)


class SqueezeExcite(nn.Module):
    def __init__(self: "SqueezeExcite", dim: int, dim_out: int, reduction: int = 2, dim_minimum: int = 8) -> None:
        super().__init__()
        inner_dim = max(dim_minimum, dim_out // reduction)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.layers = nn.Sequential(
            nn.Conv1d(dim, inner_dim, 1),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim_out, 1),
            nn.Sigmoid(),
        )

    def forward_body(self: "SqueezeExcite", x: torch.Tensor) -> torch.Tensor:
        avg_pool = self.global_avg_pool(x)
        return self.layers(avg_pool)

    def forward(self: "SqueezeExcite", x: torch.Tensor) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("SqueezeExcite")
        with context_manager:
            return self.forward_body(x)


class ResidualUnit(nn.Module):
    def __init__(
        self: "ResidualUnit",
        dim_in: int,
        dim_out: int,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(1, dim_out) if norm else nn.Identity()
        self.activation = nn.SiLU()

    def forward_body(self: "ResidualUnit", x: torch.Tensor, scale_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return x

    def forward(self: "ResidualUnit", x: torch.Tensor, scale_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("ResidualUnit")
        with context_manager:
            return self.forward_body(x, scale_shift)


class ResidualBlock(nn.Module):
    def __init__(
        self: "ResidualBlock",
        dim_in: int,
        dim_out: int,
        dim_time: Optional[int] = None,
        dim_cond: Optional[int] = None,
        use_gca: bool = False,
    ) -> None:
        super().__init__()
        self.has_time_cond = dim_time is not None
        self.has_cond = dim_cond is not None

        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(int(dim_time) + int(dim_cond), dim_out * 2),
            )
            if dim_time or dim_cond
            else None
        )
        self.block1 = ResidualUnit(dim_in, dim_out)
        self.block2 = ResidualUnit(dim_out, dim_out)

        self.res_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.se = GlobalContext(dim_out, dim_out) if use_gca else SqueezeExcite(dim_out, dim_out)

    def forward(
        self: "ResidualBlock",
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and (self.has_time_cond or self.has_cond):
            cond_emb = tuple(filter(lambda tensor: tensor is not None, (t, c)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, "b c -> b c 1")
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        h = h * self.se(h)

        return h + self.res_conv(x)
