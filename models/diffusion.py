from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812

from library.osu.from_beatmap import AUDIO_DIM, TOTAL_DIM
from library.scheduler import CosineBetaScheduler, StridedBetaScheduler
from modules.unet import UNet

MINIMUM_LENGTH = 1024


class OsuFusion(nn.Module):
    def __init__(
        self: "OsuFusion",
        dim_h: int,
        dim_cond: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        dim_learned_sinu: int = 16,
        res_strides: Tuple[int] = (2, 2, 2, 2),
        res_dilations: Tuple[int] = (1, 3, 9),
        attn_dim_head: int = 32,
        attn_heads: int = 8,
        attn_depth: int = 4,
        attn_dropout: float = 0.25,
        attn_sdpa: bool = True,
        attn_ff_dropout: float = 0.25,
        timesteps: int = 1000,
        sample_steps: int = 35,
    ) -> None:
        super().__init__()

        self.unet = UNet(
            TOTAL_DIM + AUDIO_DIM,
            TOTAL_DIM,
            dim_h,
            dim_cond,
            dim_h_mult,
            dim_learned_sinu,
            res_strides,
            res_dilations,
            attn_dim_head,
            attn_heads,
            attn_depth,
            attn_dropout,
            attn_sdpa,
            attn_ff_dropout,
        )

        self.scheduler = CosineBetaScheduler(timesteps, self.unet)
        self.sampling_scheduler = StridedBetaScheduler(self.scheduler, sample_steps, self.unet)
        self.depth = len(dim_h_mult) - 1

    def pad_data(self: "OsuFusion", x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        original_length = x.shape[-1]
        target_length = max(MINIMUM_LENGTH, ((x.shape[-1] - 1) // (2**self.depth) + 1) * (2**self.depth))
        pad_size = target_length - x.shape[-1]
        x = F.pad(x, (0, pad_size))
        return x, (..., slice(0, original_length))

    def forward(
        self: "OsuFusion",
        a: torch.Tensor,
        c: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        sample_steps: Optional[int] = None,
    ) -> torch.Tensor:
        if sample_steps is not None:
            scheduler = StridedBetaScheduler(self.scheduler, sample_steps, self.unet)
        else:
            scheduler = self.sampling_scheduler

        a, _slice = self.pad_data(a)
        return scheduler.sample(a, c, x)[_slice]
