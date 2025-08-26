from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from einops import repeat
from torch.nn import functional as F
from tqdm.auto import tqdm

from osu_fusion.data.const import AUDIO_DIM, BEATMAP_DIM, CONTEXT_DIM
from osu_fusion.models.backbone.unet import UNet


class OsuFusionUNet(nn.Module):
    def __init__(
        self: "OsuFusionUNet",
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        dim_t: int = 256,
        num_layer_blocks: Tuple[int] = (2, 2, 2, 2),
        num_middle_transformers: int = 2,
        attn_dim_head: int = 64,
        attn_context_len: int = 4096,
        cond_drop_prob: float = 0.5,
        train_timesteps: int = 1000,
        sampling_timesteps: int = 35,
    ) -> None:
        super().__init__()

        self.unet = UNet(
            dim_in_x=BEATMAP_DIM,
            dim_in_a=AUDIO_DIM,
            dim_in_c=CONTEXT_DIM,
            dim_h=dim_h,
            dim_h_mult=dim_h_mult,
            dim_t=dim_t,
            num_layer_blocks=num_layer_blocks,
            num_middle_transformers=num_middle_transformers,
            attn_dim_head=attn_dim_head,
            attn_context_len=attn_context_len,
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
    def trainable_params(self: "OsuFusionUNet") -> Tuple[nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)

    def set_full_bf16(self: "OsuFusionUNet") -> None:
        self.unet = self.unet.bfloat16()

    @torch.inference_mode()
    def sample(
        self: "OsuFusionUNet",
        n: int,
        a_lat: torch.Tensor,
        a_lat_intermediates: Optional[torch.Tensor],
        c_prep: torch.Tensor,
        c_uncond_prep: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        cond_scale: float = 2.0,
    ) -> torch.Tensor:
        assert cond_scale == 1.0 or c_uncond_prep is not None, "If cond_scale is not 1.0, c_uncond can't be None"

        b, device = a_lat.shape[0], a_lat.device
        if x is None:
            x = torch.randn((b, BEATMAP_DIM, n), device=device)
        x *= self.sampling_scheduler.init_noise_sigma

        self.sampling_scheduler.set_timesteps(self.sampling_timesteps)
        for t in tqdm(self.sampling_scheduler.timesteps, desc="sampling loop time step", dynamic_ncols=True):
            t_batched = repeat(t, "... -> b ...", b=b).long().to(device)
            x_scaled = self.sampling_scheduler.scale_model_input(x, t)
            pred = self.unet.forward_with_cond_scale(
                x_scaled,
                a_lat,
                a_lat_intermediates,
                t_batched,
                c_prep,
                c_uncond_prep,
                cond_scale=cond_scale,
            )
            x = self.sampling_scheduler.step(pred, t, x).prev_sample

        return x

    def forward(
        self: "OsuFusionUNet",
        x: torch.Tensor,
        a_lat: torch.Tensor,
        a_lat_intermediates: Optional[torch.Tensor],
        c_prep: torch.Tensor,
        orig_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(x, device=x.device)
        timesteps = torch.randint(
            0,
            self.train_scheduler.config.num_train_timesteps,
            (x.shape[0],),
            dtype=torch.int64,
            device=x.device,
        )
        x_noisy = self.train_scheduler.add_noise(x, noise, timesteps)

        pred = self.unet(x_noisy, a_lat, a_lat_intermediates, timesteps, c_prep)

        # Calculate loss
        v_target = self.train_scheduler.get_velocity(x, noise, timesteps)
        loss = F.mse_loss(pred, v_target, reduction="none")

        # Create mask for losses to ignore padding
        if orig_lens is not None:
            b, _, n = x.shape
            mask = torch.ones((b, n), device=x.device)
            for i, orig in enumerate(orig_lens):
                mask[i, orig:] = 0.0
            mask = repeat(mask, "b n -> b d n", d=BEATMAP_DIM)
            return (loss * mask).sum() / mask.sum()
        return loss.mean()
