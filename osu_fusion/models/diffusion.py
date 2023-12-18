from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, reduce
from torch.nn import functional as F  # noqa: N812
from tqdm.auto import tqdm

from osu_fusion.library.osu.from_beatmap import AUDIO_DIM, CONTEXT_DIM, TOTAL_DIM
from osu_fusion.modules.scheduler import GaussianDiffusionContinuousTimes
from osu_fusion.modules.unet import UNet

MINIMUM_LENGTH = 1024


def right_pad_dims_to(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


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
        timesteps: int = 35,
        min_snr_gamma: int = 5,
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
        )

        self.scheduler = GaussianDiffusionContinuousTimes(timesteps=timesteps)
        self.depth = len(dim_h_mult)
        self.min_snr_gamma = min_snr_gamma
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

        timesteps = self.scheduler.get_sampling_timesteps(b, device=device)
        for t, t_next in tqdm(timesteps, desc="sampling loop time step"):
            x = self.p_sample(x, a, t, c, t_next=t_next)

        return x[_slice]

    def forward(
        self: "OsuFusion",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(x)

        x_noisy, log_snr, _, _ = self.scheduler.q_sample(x, t, noise=noise)
        noise_cond = self.scheduler.get_condition(t)

        pred = self.unet(x_noisy, a, noise_cond, c)
        target = noise

        losses = F.mse_loss(pred, target, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")

        snr = log_snr.exp()
        clamped_snr = snr.clone().clamp_(max=self.min_snr_gamma)
        loss_weight = clamped_snr / snr

        losses = losses * loss_weight
        return losses.mean()
