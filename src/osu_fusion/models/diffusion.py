from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from einops import repeat
from torch.nn import functional as F
from tqdm.auto import tqdm

from osu_fusion.data.const import AUDIO_DIM, BEATMAP_DIM, CONTEXT_DIM
from osu_fusion.models.backbone.unet import UNet


class OsuFusion(nn.Module):
    def __init__(
        self: "OsuFusion",
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 4),
        num_layer_blocks: Tuple[int] = (3, 3, 3, 3),
        num_middle_transformers: int = 3,
        attn_dim_head: int = 64,
        attn_heads: int = 16,
        attn_kv_heads: int = 1,
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
            num_layer_blocks=num_layer_blocks,
            num_middle_transformers=num_middle_transformers,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len,
        )

        self.scheduler = DDIMScheduler(
            num_train_timesteps=train_timesteps,
            beta_schedule="linear",
            set_alpha_to_one=False,
            thresholding=True,
            dynamic_thresholding_ratio=0.995,  # Allow a little value to exceed the clipping threshold
        )
        self.train_timesteps = train_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.cond_drop_prob = cond_drop_prob

    @property
    def trainable_params(self: "OsuFusion") -> Tuple[nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)

    def set_full_bf16(self: "OsuFusion") -> None:
        self.unet = self.unet.bfloat16()

    @torch.inference_mode()
    def sample(
        self: "OsuFusion",
        n: int,
        a_lat: torch.Tensor,
        c_prep: torch.Tensor,
        c_uncond_prep: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        cond_scale: float = 2.0,
    ) -> torch.Tensor:
        assert cond_scale == 1.0 or c_uncond_prep is not None, "If cond_scale is not 1.0, c_uncond can't be None"

        b, device = a_lat.shape[0], a_lat.device
        if x is None:
            x = torch.randn((b, BEATMAP_DIM, n), device=device)

        self.scheduler.set_timesteps(self.sampling_timesteps)
        for t in tqdm(self.scheduler.timesteps, desc="sampling loop time step", dynamic_ncols=True):
            t_batched = repeat(t, "... -> b ...", b=b).long().to(device)
            pred = self.unet.forward_with_cond_scale(x, a_lat, t_batched, c_prep, c_uncond_prep, cond_scale=cond_scale)
            x = self.scheduler.step(pred, t, x).prev_sample

        return x

    def forward(
        self: "OsuFusion",
        x: torch.Tensor,
        a_lat: torch.Tensor,
        c_prep: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(x, device=x.device)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (x.shape[0],),
            dtype=torch.int64,
            device=x.device,
        )
        x_noisy = self.scheduler.add_noise(x, noise, timesteps)

        pred = self.unet(x_noisy, a_lat, timesteps, c_prep)

        # Calculate loss
        loss = F.mse_loss(pred, noise, reduction="none")
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
