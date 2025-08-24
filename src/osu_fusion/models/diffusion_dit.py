from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from einops import repeat
from torch.nn import functional as F
from tqdm.auto import tqdm

from osu_fusion.data.const import AUDIO_DIM, BEATMAP_DIM, CONTEXT_DIM
from osu_fusion.models.backbone.dit import DiT


class OsuFusionDiT(nn.Module):
    def __init__(
        self: "OsuFusionDiT",
        dim_h: int,
        dim_h_mult: int = 4,
        dim_t: int = 256,
        patch_size: int = 8,
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
            patch_size=patch_size,
            depth=depth,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len // patch_size,
        )

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=train_timesteps,
            prediction_type="v_prediction",
            clip_sample=False,
            rescale_betas_zero_snr=True,
        )
        self.sampling_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=train_timesteps,
            prediction_type="v_prediction",
            algorithm_type="sde-dpmsolver++",
        )
        self.train_timesteps = train_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.cond_drop_prob = cond_drop_prob

    @property
    def trainable_params(self: "OsuFusionDiT") -> Tuple[nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)

    def set_full_bf16(self: "OsuFusionDiT") -> None:
        self.dit = self.dit.bfloat16()

    @torch.inference_mode()
    def sample(
        self: "OsuFusionDiT",
        n: int,
        a_patch: torch.Tensor,
        a_mlp_out: torch.Tensor,
        c: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        cond_scale: float = 2.0,
    ) -> torch.Tensor:
        b, device = a_patch.shape[0], a_patch.device
        if x is None:
            x = torch.randn((b, BEATMAP_DIM, n), device=device)
        x *= self.sampling_scheduler.init_noise_sigma

        self.sampling_scheduler.set_timesteps(self.sampling_timesteps)
        for t in tqdm(self.sampling_scheduler.timesteps, desc="sampling loop time step", dynamic_ncols=True):
            t_batched = repeat(t, "... -> b ...", b=b).long().to(device)
            x_scaled = self.sampling_scheduler.scale_model_input(x, t)
            pred = self.dit.forward_with_cond_scale(
                x_scaled,
                a_patch,
                a_mlp_out,
                t_batched,
                c,
                cond_scale=cond_scale,
            )
            x = self.sampling_scheduler.step(pred, t, x).prev_sample

        return x

    def forward(
        self: "OsuFusionDiT",
        x: torch.Tensor,
        a_patch: torch.Tensor,
        a_mlp_out: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(x, device=x.device)
        timesteps = torch.randint(
            0,
            self.sampling_scheduler.config.num_train_timesteps,
            (x.shape[0],),
            dtype=torch.int64,
            device=x.device,
        )
        x_noisy = self.sampling_scheduler.add_noise(x, noise, timesteps)

        pred = self.dit(x_noisy, a_patch, a_mlp_out, timesteps, c)

        # Calculate loss
        v_target = self.train_scheduler.get_velocity(x, noise, timesteps)
        loss = F.mse_loss(pred, v_target, reduction="none")
        return loss.mean()

        # # Create mask for losses to ignore padding
        # if orig_len is not None:
        #     b, _, n = x.shape
        #     mask = torch.ones((b, n), device=x.device)
        #     for i, orig in enumerate(orig_len):
        #         mask[i, orig:] = 0.0
        #     mask = repeat(mask, "b n -> b d n", d=BEATMAP_LATENT_DIM)
        #     return (loss * mask).sum() / mask.sum()
        # return loss.mean()
