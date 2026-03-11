from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from einops import repeat
from torch.nn import functional as F
from tqdm.auto import tqdm

from osu_fusion.data.const import AUDIO_DIM, BEATMAP_DIM, CONTEXT_DIM
from osu_fusion.models.backbone.dit import DiT, LengthPredictor


@dataclass
class DiTConfig:
    dim_h: int = 384
    dim_h_mult: int = 4
    dim_t: int = 256
    audio_patch_size: int = 8
    depth: int = 12
    attn_dim_head: int = 64
    attn_heads: int = 6
    attn_kv_heads: int = 3
    attn_context_len: int = 4096
    cond_drop_prob: float = 0.5
    train_timesteps: int = 1000
    sampling_timesteps: int = 35


DiTConfig_S = DiTConfig()
DiTConfig_M = DiTConfig(dim_h=512, depth=16, attn_heads=8, attn_kv_heads=4)
DiTConfig_L = DiTConfig(dim_h=768, depth=24, attn_heads=12, attn_kv_heads=6)


class OsuFusionDiT(nn.Module):
    def __init__(
        self: "OsuFusionDiT",
        dim_h: int,
        dim_h_mult: int = 4,
        dim_t: int = 256,
        audio_patch_size: int = 8,
        depth: int = 24,
        attn_dim_head: int = 64,
        attn_heads: int = 16,
        attn_kv_heads: int = 8,
        attn_context_len: int = 4096,
        cond_drop_prob: float = 0.5,
        train_timesteps: int = 1000,
        sampling_timesteps: int = 35,
    ) -> None:
        super().__init__()

        self.dit = DiT(
            dim_in_x=BEATMAP_DIM,
            dim_in_a=AUDIO_DIM,
            dim_in_c=CONTEXT_DIM,
            dim_t=dim_t,
            dim_h=dim_h,
            dim_h_mult=dim_h_mult,
            audio_patch_size=audio_patch_size,
            depth=depth,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len,
        )
        self.length_predictor = LengthPredictor(
            dim_a=AUDIO_DIM,
            dim_c=CONTEXT_DIM,
        )

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=train_timesteps,
            prediction_type="v_prediction",
            clip_sample=False,
            rescale_betas_zero_snr=True,
            thresholding=True,
        )
        self.sampling_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=train_timesteps,
            prediction_type="v_prediction",
            algorithm_type="sde-dpmsolver++",
            thresholding=True,
        )
        self.train_timesteps = train_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.cond_drop_prob = cond_drop_prob

    @property
    def trainable_params(self: "OsuFusionDiT") -> Tuple[nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)

    def set_full_bf16(self: "OsuFusionDiT") -> None:
        self.dit = self.dit.bfloat16()
        self.length_predictor = self.length_predictor.bfloat16()

    def predict_length(self: "OsuFusionDiT", a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.length_predictor(a, c)

    @torch.inference_mode()
    def sample(
        self: "OsuFusionDiT",
        a: torch.Tensor,
        c: torch.Tensor,
        n: Optional[int] = None,
        x: Optional[torch.Tensor] = None,
        cond_scale: float = 2.0,
    ) -> torch.Tensor:
        b, device = a.shape[0], a.device

        if n is None:
            pred_log_len = self.predict_length(a, c)
            n = int(pred_log_len.exp().clamp(min=1).round().item())

        if x is None:
            x = torch.randn((b, n, BEATMAP_DIM), device=device)
        x *= self.sampling_scheduler.init_noise_sigma

        self.sampling_scheduler.set_timesteps(self.sampling_timesteps)
        for t in tqdm(self.sampling_scheduler.timesteps, desc="sampling loop time step", dynamic_ncols=True):
            t_batched = repeat(t, "... -> b ...", b=b).long().to(device)
            x_scaled = self.sampling_scheduler.scale_model_input(x, t)
            pred = self.dit.forward_with_cond_scale(
                x_scaled,
                a,
                t_batched,
                c,
                cond_scale=cond_scale,
            )
            x = self.sampling_scheduler.step(pred, t, x).prev_sample

        return x

    def forward(
        self: "OsuFusionDiT",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
        orig_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Length prediction loss (in log space to match diffusion loss scale)
        pred_log_len = self.length_predictor(a, c)
        if orig_lens is not None:
            target_len = orig_lens.float()
        else:
            target_len = torch.full((x.shape[0],), x.shape[1], device=x.device, dtype=x.dtype)
        target_log_len = torch.log(target_len.clamp(min=1))
        length_loss = F.mse_loss(pred_log_len, target_log_len)

        # Diffusion loss
        noise = torch.randn_like(x, device=x.device)
        timesteps = torch.randint(
            0,
            self.train_scheduler.config.num_train_timesteps,
            (x.shape[0],),
            dtype=torch.int64,
            device=x.device,
        )
        x_noisy = self.train_scheduler.add_noise(x, noise, timesteps)

        pred = self.dit(x_noisy, a, timesteps, c, self.cond_drop_prob)

        v_target = self.train_scheduler.get_velocity(x, noise, timesteps)
        diff_loss = F.mse_loss(pred, v_target, reduction="none")

        if orig_lens is not None:
            b, n, _ = x.shape
            mask = torch.ones((b, n), device=x.device)
            for i, orig in enumerate(orig_lens):
                mask[i, orig:] = 0.0
            mask = repeat(mask, "b n -> b n d", d=BEATMAP_DIM)
            diff_loss = (diff_loss * mask).sum() / mask.sum()
        else:
            diff_loss = diff_loss.mean()

        return diff_loss, length_loss
