from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from einops import reduce, repeat
from torch.nn import functional as F  # noqa: N812
from tqdm.auto import tqdm

from osu_fusion.library.osu.data.encode import TOTAL_DIM
from osu_fusion.modules.mmdit import MMDiT
from osu_fusion.scripts.dataset_creator import AUDIO_DIM, CONTEXT_DIM

MINIMUM_LENGTH = 1024


class OsuFusion(nn.Module):
    def __init__(
        self: "OsuFusion",
        dim_h: int,
        dim_h_mult: int = 4,
        depth: int = 12,
        cross_embed_kernel_sizes: Tuple[int] = (3, 7, 15),
        num_register_tokens: int = 4,
        attn_heads: int = 6,
        attn_qk_norm: bool = True,
        attn_one_kv: bool = True,
        attn_causal: bool = True,
        attn_use_rotary_emb: bool = True,
        cond_drop_prob: float = 0.25,
        train_timesteps: int = 1000,
        sampling_timesteps: int = 35,
        dynamic_thresholding_percentile: float = 0.995,
    ) -> None:
        super().__init__()

        self.mmdit = MMDiT(
            dim_in_x=TOTAL_DIM,
            dim_in_a=AUDIO_DIM,
            dim_in_c=CONTEXT_DIM - 1,
            dim_h=dim_h,
            dim_h_mult=dim_h_mult,
            depth=depth,
            cross_embed_kernel_sizes=cross_embed_kernel_sizes,
            num_register_tokens=num_register_tokens,
            attn_heads=attn_heads,
            attn_qk_norm=attn_qk_norm,
            attn_one_kv=attn_one_kv,
            attn_causal=attn_causal,
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
        self.depth = depth

    @torch.no_grad()
    def sample(
        self: "OsuFusion",
        a: torch.Tensor,
        c: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        cond_scale: float = 7.0,
    ) -> torch.Tensor:
        (b, _, n), device = a.shape, a.device
        if x is None:
            x = torch.randn((b, TOTAL_DIM, n), device=device)

        self.scheduler.set_timesteps(self.sampling_timesteps)
        for t in tqdm(self.scheduler.timesteps, desc="sampling loop time step"):
            t_batched = repeat(t, "... -> b ...", b=b).long().to(device)
            pred = self.mmdit.forward_with_cond_scale(x, a, t_batched, c, cond_scale=cond_scale)
            x = self.scheduler.step(pred, t, x).prev_sample

        return x

    def forward(
        self: "OsuFusion",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
        orig_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert x.shape[-1] == a.shape[-1], "x and a must have the same number of sequence length"
        # Remove the last element of the context tensor (I don't want to include bpm)
        c = c[:, :-1]

        noise = torch.randn_like(x)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (x.shape[0],),
            dtype=torch.int64,
            device=x.device,
        )
        x_noisy = self.scheduler.add_noise(x, noise, timesteps)

        pred = self.mmdit(x_noisy, a, timesteps, c, self.cond_drop_prob)

        losses = F.mse_loss(pred, noise, reduction="none")
        if orig_len is not None:
            for i, orig in enumerate(orig_len):
                losses[i, orig:] = 0.0

        losses = reduce(losses, "b ... -> b", "mean")
        return losses.mean()
