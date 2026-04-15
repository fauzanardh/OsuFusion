import copy
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from einops import repeat
from torch.nn import functional as F
from tqdm.auto import tqdm

from osu_fusion.data.descriptors import NUM_DESCRIPTORS
from osu_fusion.data.encode import SEQ_DIM
from osu_fusion.models.backbone.dit import DiT


@dataclass
class FlowDiTConfig:
    dim_h: int = 384
    dim_h_mult: int = 6
    dim_t: int = 256
    dim_cond_fourier: int = 32
    beatmap_patch_size: int = 4
    audio_patch_size: int = 4
    mmdit_depth: int = 9
    dit_depth: int = 3
    attn_dim_head: int = 64
    attn_heads: int = 6
    attn_context_len: int = 8192
    cond_drop_prob: float = 0.2
    train_timesteps: int = 1000
    sampling_timesteps: int = 50
    num_descriptors: int = NUM_DESCRIPTORS
    num_mappers: int = 0


FlowDiTConfig_S = FlowDiTConfig()
FlowDiTConfig_M = FlowDiTConfig(dim_h=512, mmdit_depth=12, dit_depth=4, attn_heads=8)
FlowDiTConfig_L = FlowDiTConfig(dim_h=768, mmdit_depth=24, dit_depth=8, attn_heads=12)


class OsuFusionFlowDiT(nn.Module):
    def __init__(
        self: "OsuFusionFlowDiT",
        dim_h: int = 384,
        dim_h_mult: int = 6,
        dim_t: int = 256,
        dim_cond_fourier: int = 32,
        beatmap_patch_size: int = 4,
        audio_patch_size: int = 4,
        mmdit_depth: int = 9,
        dit_depth: int = 3,
        attn_dim_head: int = 64,
        attn_heads: int = 6,
        attn_context_len: int = 8192,
        cond_drop_prob: float = 0.2,
        train_timesteps: int = 1000,
        sampling_timesteps: int = 50,
        num_descriptors: int = 0,
        num_mappers: int = 0,
        weighting_scheme: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 1.29,
    ) -> None:
        super().__init__()

        self.dit = DiT(
            dim_t=dim_t,
            dim_h=dim_h,
            dim_h_mult=dim_h_mult,
            dim_cond_fourier=dim_cond_fourier,
            beatmap_patch_size=beatmap_patch_size,
            audio_patch_size=audio_patch_size,
            mmdit_depth=mmdit_depth,
            dit_depth=dit_depth,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_context_len=attn_context_len,
            num_descriptors=num_descriptors,
            num_mappers=num_mappers,
        )

        self.sampling_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=train_timesteps,
        )
        # Deep copy for training — never modified by set_timesteps during sampling
        self.noise_scheduler = copy.deepcopy(self.sampling_scheduler)
        self.noise_scheduler.set_timesteps(train_timesteps)

        self.train_timesteps = train_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.cond_drop_prob = cond_drop_prob
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale

    @property
    def trainable_params(self: "OsuFusionFlowDiT") -> Iterator[nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)

    def set_full_bf16(self: "OsuFusionFlowDiT") -> None:
        self.dit = self.dit.bfloat16()

    @torch.inference_mode()
    def sample(
        self: "OsuFusionFlowDiT",
        a: torch.Tensor,
        c: torch.Tensor,
        descriptors: Optional[torch.Tensor] = None,
        mappers: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        cond_scale: float = 2.0,
    ) -> torch.Tensor:
        b, n, device = a.shape[0], a.shape[1], a.device

        if x is None:
            x = torch.randn((b, n, SEQ_DIM), device=device)

        self.sampling_scheduler.set_timesteps(self.sampling_timesteps)
        for t in tqdm(self.sampling_scheduler.timesteps, desc="sampling loop time step", dynamic_ncols=True):
            t_batched = repeat(t, "... -> b ...", b=b).to(device)
            pred = self.dit.forward_with_cond_scale(
                x,
                a,
                t_batched,
                c,
                descriptors=descriptors,
                mappers=mappers,
                cond_scale=cond_scale,
            )
            x = self.sampling_scheduler.step(pred, t, x).prev_sample

        return x

    def forward(
        self: "OsuFusionFlowDiT",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
        descriptors: Optional[torch.Tensor] = None,
        mappers: Optional[torch.Tensor] = None,
        orig_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = x.shape[0]
        noise = torch.randn_like(x, device=x.device)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=b,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        indices = indices.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)
        timesteps = self.noise_scheduler.timesteps[indices].to(device=x.device)

        sigmas = self.noise_scheduler.sigmas[indices].to(device=x.device, dtype=x.dtype)
        sigmas_expand = sigmas.view(-1, 1, 1)
        x_noisy = (1.0 - sigmas_expand) * x + sigmas_expand * noise

        pred_v = self.dit(
            x_noisy,
            a,
            timesteps,
            c,
            descriptors=descriptors,
            mappers=mappers,
            cond_drop_prob=self.cond_drop_prob,
            orig_lens=orig_lens,
        )
        target_v = noise - x
        fm_loss = F.mse_loss(pred_v, target_v, reduction="none")

        if orig_lens is not None:
            n = x.shape[1]
            idx = torch.arange(n, device=x.device).unsqueeze(0)
            mask = (idx < orig_lens.unsqueeze(1)).float()
            mask = repeat(mask, "b n -> b n d", d=SEQ_DIM)
            per_sample_loss = (fm_loss * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
            fm_loss = per_sample_loss.mean()
        else:
            fm_loss = fm_loss.mean()

        return fm_loss
