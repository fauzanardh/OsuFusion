import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F  # noqa: N812
from tqdm.auto import tqdm

from osu_fusion.library.osu.data.encode import TOTAL_DIM
from osu_fusion.library.scheduler import EDMScheduler
from osu_fusion.modules.mmdit import MMDiT
from osu_fusion.scripts.dataset_creator import AUDIO_DIM, CONTEXT_DIM


class OsuFusion(nn.Module):
    def __init__(
        self: "OsuFusion",
        dim_h: int,
        dim_h_mult: int = 4,
        patch_size: int = 4,
        depth: int = 12,
        attn_heads: int = 16,
        attn_dim_head: int = 64,
        attn_kv_heads: int = 4,
        attn_qk_norm: bool = True,
        attn_causal: bool = False,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 8192,
        attn_infini: bool = True,
        attn_segment_len: int = 1024,
        cond_drop_prob: float = 0.5,
        sampling_timesteps: int = 35,
    ) -> None:
        super().__init__()

        self.mmdit = MMDiT(
            dim_in_x=TOTAL_DIM,
            dim_in_a=AUDIO_DIM,
            dim_in_c=CONTEXT_DIM,
            dim_h=dim_h,
            dim_h_mult=dim_h_mult,
            patch_size=patch_size,
            depth=depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_kv_heads=attn_kv_heads,
            attn_qk_norm=attn_qk_norm,
            attn_causal=attn_causal,
            attn_use_rotary_emb=attn_use_rotary_emb,
            attn_context_len=attn_context_len,
            attn_infini=attn_infini,
            attn_segment_len=attn_segment_len,
        )
        self.scheduler = EDMScheduler()
        self.sampling_timesteps = sampling_timesteps
        self.cond_drop_prob = cond_drop_prob

    def set_full_bf16(self: "OsuFusion") -> None:
        self.mmdit = self.mmdit.bfloat16()

    @torch.inference_mode()
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

        sigmas = self.scheduler.sample_schedule(self.sampling_timesteps, device)
        gammas = torch.where(
            (sigmas >= self.scheduler.s_tmin) & (sigmas <= self.scheduler.s_tmax),
            min(self.scheduler.s_churn / self.sampling_timesteps, math.sqrt(2) - 1),
            0.0,
        )
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))
        x = x * sigmas[0]
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc="sampling loop time step", dynamic_ncols=True):
            sigma, sigma_next, gamma = [t.item() for t in (sigma, sigma_next, gamma)]

            eps = self.scheduler.s_noise * torch.randn_like(x, device=device)

            sigma_hat = sigma + gamma * sigma
            added_noise = math.sqrt(sigma_hat**2 - sigma**2) * eps
            x_noisy = x + added_noise

            model_output = self.scheduler.preconditioned_network_forward(
                self.mmdit.forward_with_cond_scale,
                x_noisy,
                a,
                sigma_hat,
                c,
                cond_scale=cond_scale,
            )

            x_denoised = (x_noisy - model_output) / sigma_hat
            x_next = x_noisy + (sigma_next - sigma_hat) * x_denoised

            # second order correction (heun's method)
            if sigma_next != 0:
                model_output_next = self.scheduler.preconditioned_network_forward(
                    self.mmdit.forward_with_cond_scale,
                    x_next,
                    a,
                    sigma_next,
                    c,
                    cond_scale=cond_scale,
                )

                x_prime_denoised = (x_next - model_output_next) / sigma_next
                x_next = x_noisy + 0.5 * (sigma_next - sigma_hat) * (x_denoised + x_prime_denoised)

            x = x_next

        x = x.clamp(-1.0, 1.0)
        return x

    def forward(
        self: "OsuFusion",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
        orig_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert x.shape[-1] == a.shape[-1], "x and a must have the same number of sequence length"

        sigmas = self.scheduler.noise_distribution(x.shape[0], x.device)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        noise = torch.randn_like(x, device=x.device)
        x_noisy = x + padded_sigmas * noise

        pred_x0 = self.scheduler.preconditioned_network_forward(
            self.mmdit.forward,
            x_noisy,
            a,
            sigmas,
            c,
            cond_drop_prob=self.cond_drop_prob,
        )

        # Calculate loss
        loss = F.mse_loss(pred_x0, x, reduction="none")

        # Create mask for losses to ignore padding
        b, _, n = x.shape
        mask = torch.ones((b, n), device=x.device)
        if orig_len is not None:
            for i, orig in enumerate(orig_len):
                mask[i, orig:] = 0.0
        mask = repeat(mask, "b n -> b d n", d=TOTAL_DIM)
        return (loss * mask).sum() / mask.sum()
