import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
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

    def forward_body(
        self: "ResidualUnit",
        x: torch.Tensor,
        scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return x

    def forward(
        self: "ResidualUnit",
        x: torch.Tensor,
        scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("ResidualUnit")
        with context_manager:
            return self.forward_body(x, scale_shift)


class ResidualBlock(nn.Module):
    def __init__(
        self: "ResidualBlock",
        dim_in: int,
        dim_out: int,
        dim_audio: Optional[int] = None,
        dim_time: Optional[int] = None,
        dim_cond: Optional[int] = None,
        use_gca: bool = False,
    ) -> None:
        super().__init__()
        total_global_cond_dim = int(dim_time or 0) + int(dim_cond or 0)
        self.global_mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(total_global_cond_dim, 2 * dim_out),
            )
            if total_global_cond_dim > 0
            else None
        )
        self.audio_conv = (
            nn.Sequential(
                nn.SiLU(),
                nn.Conv1d(dim_audio, 2 * dim_out, 3, padding=1),
            )
            if dim_audio is not None and dim_audio > 0
            else None
        )

        self.block1 = ResidualUnit(dim_in, dim_out, norm=True)
        self.block2 = ResidualUnit(dim_out, dim_out, norm=True)
        self.res_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.se = GlobalContext(dim_out, dim_out) if use_gca else SqueezeExcite(dim_out, dim_out)

    def forward(
        self: "ResidualBlock",
        x: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale, shift = 0.0, 0.0

        if self.global_mlp is not None:
            global_cond_emb = tuple(filter(lambda t: t is not None, (t, c)))
            if global_cond_emb:
                global_cond_emb = torch.cat(global_cond_emb, dim=-1)
                global_cond_emb = self.global_mlp(global_cond_emb)
                global_cond_emb = rearrange(global_cond_emb, "b d -> b d 1")
                scale_global, shift_global = global_cond_emb.chunk(2, dim=1)
                scale = scale + scale_global
                shift = shift + shift_global

        if self.audio_conv is not None and a is not None:
            if a.shape[-1] != x.shape[-1]:
                a = F.interpolate(a, size=x.shape[-1], mode="nearest", align_corners=False)

            audio_cond_emb = self.audio_conv(a)
            scale_local, shift_local = audio_cond_emb.chunk(2, dim=1)
            scale = scale + scale_local
            shift = shift + shift_local

        h = self.block1(
            x,
            scale_shift=(scale, shift)
            if (self.global_mlp is not None or self.audio_conv is not None)
            else None,  # Janky check
        )
        h = self.block2(h)
        h = h * self.se(h)
        return h + self.res_conv(x)
