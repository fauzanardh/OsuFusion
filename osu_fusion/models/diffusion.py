from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from einops import reduce, repeat
from torch.nn import functional as F  # noqa: N812
from tqdm.auto import tqdm

from osu_fusion.library.osu.from_beatmap import AUDIO_DIM, CONTEXT_DIM, TOTAL_DIM
from osu_fusion.modules.unet import UNet

MINIMUM_LENGTH = 1024


class OsuFusion(nn.Module):
    def __init__(
        self: "OsuFusion",
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        res_strides: Tuple[int] = (2, 2, 2, 2),
        res_num_layers: int = 4,
        attn_dim_head: int = 32,
        attn_heads: int = 8,
        attn_depth: int = 4,
        attn_dropout: float = 0.1,
        attn_use_global_context_attention: bool = True,
        attn_sdpa: bool = True,
        attn_use_rotary_emb: bool = True,
        train_timesteps: int = 1000,
        sampling_timesteps: int = 35,
        dynamic_thresholding_percentile: float = 0.995,
    ) -> None:
        super().__init__()

        self.unet = UNet(
            TOTAL_DIM + AUDIO_DIM,
            TOTAL_DIM,
            dim_h,
            CONTEXT_DIM,
            dim_h_mult=dim_h_mult,
            res_strides=res_strides,
            res_num_layers=res_num_layers,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_depth=attn_depth,
            attn_dropout=attn_dropout,
            attn_use_global_context_attention=attn_use_global_context_attention,
            attn_sdpa=attn_sdpa,
            attn_use_rotary_emb=attn_use_rotary_emb,
        )

        self.scheduler = DDPMScheduler(
            num_train_timesteps=train_timesteps,
            beta_schedule="scaled_linear",
            thresholding=True,
            dynamic_thresholding_ratio=dynamic_thresholding_percentile,
        )
        self.train_timesteps = train_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.depth = len(dim_h_mult)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

    def pad_data(self: "OsuFusion", x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        original_length = x.shape[-1]
        target_length = max(MINIMUM_LENGTH, ((x.shape[-1] - 1) // (2**self.depth) + 1) * (2**self.depth))
        pad_size = target_length - x.shape[-1]
        x = F.pad(x, (0, pad_size))
        return x, (..., slice(0, original_length))

    @torch.no_grad()
    def sample(
        self: "OsuFusion",
        a: torch.Tensor,
        c: torch.Tensor,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a, _slice = self.pad_data(a)

        (b, _, n), device = a.shape, a.device
        if x is None:
            x = torch.randn((b, TOTAL_DIM, n), device=device)
        else:
            x, _ = self.pad_data(x)

        self.scheduler.set_timesteps(self.sampling_timesteps)
        for t in tqdm(self.scheduler.timesteps, desc="sampling loop time step"):
            t_batched = repeat(t, "... -> b ...", b=b).float().to(device)
            t_batched /= self.scheduler.config.num_train_timesteps
            pred = self.unet(x, a, t_batched, c)
            x = self.scheduler.step(pred, t, x).prev_sample

        return x[_slice]

    def forward(
        self: "OsuFusion",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
        orig_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_padded, slice_ = self.pad_data(x)
        a_padded, _ = self.pad_data(a)

        assert x_padded.shape[-1] == a_padded.shape[-1], "x and a must have the same number of sequence length"

        noise = torch.randn_like(x_padded)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x_padded.shape[0],)).cuda()
        timesteps = timesteps.long()
        x_noisy = self.scheduler.add_noise(x_padded, noise, timesteps)

        t = timesteps / self.scheduler.config.num_train_timesteps
        pred = self.unet(x_noisy, a_padded, t, c)[slice_]
        target = noise[slice_]

        losses = F.mse_loss(pred, target, reduction="none")
        if orig_len is not None:
            for i, orig in enumerate(orig_len):
                losses[i, orig:] = 0.0

        losses = reduce(losses, "b ... -> b", "mean")
        return losses.mean()
