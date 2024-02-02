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
        dim_h_mult: Tuple[int] = (1, 2, 2, 4),
        num_blocks: int = 3,
        cross_embed_kernel_sizes: Tuple[int] = (3, 7, 15),
        attn_dim_head: int = 32,
        attn_heads: int = 8,
        attn_sdpa: bool = True,
        attn_use_rotary_emb: bool = True,
        cond_drop_prob: float = 0.25,
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
            num_blocks=num_blocks,
            cross_embed_kernel_sizes=cross_embed_kernel_sizes,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_sdpa=attn_sdpa,
            attn_use_rotary_emb=attn_use_rotary_emb,
        )

        self.scheduler = DDPMScheduler(
            num_train_timesteps=train_timesteps,
            beta_schedule="scaled_linear",
            # thresholding=True,
            # dynamic_thresholding_ratio=dynamic_thresholding_percentile,
        )
        self.train_timesteps = train_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.cond_drop_prob = cond_drop_prob
        self.depth = len(dim_h_mult)

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
        cond_scale: float = 7.0,
    ) -> torch.Tensor:
        a, _slice = self.pad_data(a)

        (b, _, n), device = a.shape, a.device
        if x is None:
            x = torch.randn((b, TOTAL_DIM, n), device=device)
        else:
            x, _ = self.pad_data(x)

        self.scheduler.set_timesteps(self.sampling_timesteps)
        for t in tqdm(self.scheduler.timesteps, desc="sampling loop time step"):
            t_batched = repeat(t, "... -> b ...", b=b).long().to(device)
            pred = self.unet.forward_with_cond_scale(x, a, t_batched, c, cond_scale=cond_scale)
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
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (x_padded.shape[0],),
            dtype=torch.int64,
            device=x_padded.device,
        )
        x_noisy = self.scheduler.add_noise(x_padded, noise, timesteps)

        pred = self.unet(x_noisy, a_padded, timesteps, c, self.cond_drop_prob)[slice_]
        target = noise[slice_]

        losses = F.mse_loss(pred, target, reduction="none")
        if orig_len is not None:
            for i, orig in enumerate(orig_len):
                losses[i, orig:] = 0.0

        losses = reduce(losses, "b ... -> b", "mean")
        return losses.mean()
