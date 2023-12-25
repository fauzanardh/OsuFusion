from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, reduce
from torch.nn import functional as F  # noqa: N812
from tqdm.auto import tqdm

from osu_fusion.library.osu.from_beatmap import AUDIO_DIM, CONTEXT_DIM, TOTAL_DIM
from osu_fusion.modules.scheduler import GaussianDiffusionContinuousTimes
from osu_fusion.modules.unet import UNet
from osu_fusion.modules.utils import right_pad_dims_to

MINIMUM_LENGTH = 1024


class OsuFusion(nn.Module):
    def __init__(
        self: "OsuFusion",
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        dim_learned_sinu: int = 16,
        res_strides: Tuple[int] = (2, 2, 2, 2),
        res_num_layers: int = 4,
        attn_dim_head: int = 32,
        attn_heads: int = 8,
        attn_depth: int = 4,
        attn_dropout: float = 0.1,
        attn_use_global_context_attention: bool = True,
        attn_sdpa: bool = True,
        attn_use_rotary_emb: bool = True,
        cond_drop_prob: float = 0.1,
        timesteps: int = 1000,
        sampling_steps: int = 35,
        dynamic_thresholding_percentile: float = 0.95,
    ) -> None:
        super().__init__()

        self.unet = UNet(
            TOTAL_DIM + AUDIO_DIM,
            TOTAL_DIM,
            dim_h,
            CONTEXT_DIM,
            dim_h_mult=dim_h_mult,
            dim_learned_sinu=dim_learned_sinu,
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

        self.scheduler = GaussianDiffusionContinuousTimes(timesteps=timesteps)
        self.timesteps = timesteps
        self.sampling_steps = sampling_steps
        self.cond_drop_prob = cond_drop_prob
        self.depth = len(dim_h_mult)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

    def pad_data(self: "OsuFusion", x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        original_length = x.shape[-1]
        target_length = max(MINIMUM_LENGTH, ((x.shape[-1] - 1) // (2**self.depth) + 1) * (2**self.depth))
        pad_size = target_length - x.shape[-1]
        x = F.pad(x, (0, pad_size))
        return x, (..., slice(0, original_length))

    def p_mean_variance(
        self: "OsuFusion",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.unet(x, a, self.scheduler.get_condition(t), c)
        x_0 = self.scheduler.predict_start_from_noise(x, t, pred)

        # dynamic thresholding
        s = torch.quantile(
            rearrange(x_0, "b ... -> b (...)").abs(),
            q=self.dynamic_thresholding_percentile,
            dim=-1,
        )
        s.clamp_(min=1.0)
        s = right_pad_dims_to(x_0, s)
        x_0 = x_0.clamp(-s, s) / s

        mean_and_variance = self.scheduler.q_posterior(x_0, x, t, t_next=t_next)
        return mean_and_variance

    @torch.no_grad()
    def p_sample(
        self: "OsuFusion",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x, a, t, c, t_next=t_next)
        noise = torch.randn_like(x)
        is_last_sampling_step = t_next == 0.0
        nonzero_mask = (1 - is_last_sampling_step.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred

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

        self.scheduler.timesteps = self.sampling_steps
        timesteps = self.scheduler.get_sampling_timesteps(b, device=device)
        for t, t_next in tqdm(timesteps, desc="sampling loop time step"):
            x = self.p_sample(x, a, t, c, t_next=t_next)
        self.scheduler.timesteps = self.timesteps

        return x[_slice]

    def forward(
        self: "OsuFusion",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        x_padded, slice_ = self.pad_data(x)
        a_padded, _ = self.pad_data(a)

        assert x_padded.shape[-1] == a_padded.shape[-1], "x and a must have the same number of sequence length"

        noise = torch.randn_like(x_padded)

        x_noisy, _, _, _ = self.scheduler.q_sample(x_padded, t, noise=noise)
        noise_cond = self.scheduler.get_condition(t)

        pred = self.unet(x_noisy, a_padded, noise_cond, c, self.cond_drop_prob)[slice_]
        target = noise[slice_]

        losses = F.mse_loss(pred, target, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        return losses.mean()
