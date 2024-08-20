from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from einops import rearrange, repeat
from torch.nn import functional as F  # noqa: N812
from tqdm.auto import tqdm

from osu_fusion.library.osu.data.encode import TOTAL_DIM
from osu_fusion.modules.unet import UNet
from osu_fusion.scripts.dataset_creator import AUDIO_DIM, CONTEXT_DIM


class OsuFusion(nn.Module):
    def __init__(
        self: "OsuFusion",
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        num_layer_blocks: Tuple[int] = (3, 3, 3, 3),
        num_middle_transformers: int = 3,
        cross_embed_kernel_sizes: Tuple[int] = (3, 7, 15),
        attn_dim_head: int = 64,
        attn_heads: int = 16,
        attn_kv_heads: int = 4,
        attn_qk_norm: bool = True,
        attn_causal: bool = False,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 8192,
        attn_infini: bool = True,
        attn_segment_len: int = 8192,
        patch_size: int = 4,
        cond_drop_prob: float = 0.5,
        train_timesteps: int = 1000,
        sampling_timesteps: int = 35,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.unet = UNet(
            dim_in_x=TOTAL_DIM * patch_size,
            dim_in_a=AUDIO_DIM * patch_size,
            dim_in_c=CONTEXT_DIM,
            dim_h=dim_h,
            dim_h_mult=dim_h_mult,
            num_layer_blocks=num_layer_blocks,
            num_middle_transformers=num_middle_transformers,
            cross_embed_kernel_sizes=cross_embed_kernel_sizes,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_qk_norm=attn_qk_norm,
            attn_causal=attn_causal,
            attn_use_rotary_emb=attn_use_rotary_emb,
            attn_context_len=attn_context_len,
            attn_infini=attn_infini,
            attn_segment_len=attn_segment_len,
            patch_size=patch_size,
        )

        self.scheduler = DDIMScheduler(
            num_train_timesteps=train_timesteps,
            beta_schedule="linear",
        )
        self.train_timesteps = train_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.cond_drop_prob = cond_drop_prob

    def set_full_bf16(self: "OsuFusion") -> None:
        self.unet = self.unet.bfloat16()

    def pack(self: "OsuFusion", x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b d (p n) -> b (p d) n", p=self.patch_size)

    def unpack(self: "OsuFusion", x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b (p d) n -> b d (p n)", p=self.patch_size)

    @torch.inference_mode()
    def sample(
        self: "OsuFusion",
        a: torch.Tensor,
        c: torch.Tensor,
        seed: Optional[int] = None,
        cond_scale: float = 7.0,
    ) -> torch.Tensor:
        a = self.pack(a)
        (b, _, n), d, device = a.shape, TOTAL_DIM * self.patch_size, a.device

        if seed is None:
            x = torch.randn((b, d, n), device=device)
        else:
            x = torch.randn(
                (b, d, n),
                device=device,
                generator=torch.Generator().manual_seed(seed),
            )

        self.scheduler.set_timesteps(self.sampling_timesteps)
        for t in tqdm(self.scheduler.timesteps, desc="sampling loop time step", dynamic_ncols=True):
            t_batched = repeat(t, "... -> b ...", b=b).long().to(device)
            pred = self.unet.forward_with_cond_scale(x, a, t_batched, c, cond_scale=cond_scale)
            x = self.scheduler.step(pred, t, x).prev_sample

        return self.unpack

    def forward(
        self: "OsuFusion",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
        orig_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert x.shape[-1] == a.shape[-1], "x and a must have the same number of sequence length"
        x = self.pack(x)
        a = self.pack(a)

        noise = torch.randn_like(x, device=x.device)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (x.shape[0],),
            dtype=torch.int64,
            device=x.device,
        )
        x_noisy = self.scheduler.add_noise(x, noise, timesteps)

        pred = self.unet(x_noisy, a, timesteps, c, cond_drop_prob=self.cond_drop_prob)

        # Calculate loss
        loss = F.mse_loss(pred, noise, reduction="none")

        # Create mask for losses to ignore padding
        b, d, n = x.shape
        mask = torch.ones((b, n), device=x.device)
        if orig_len is not None:
            for i, orig in enumerate(orig_len):
                mask[i, orig:] = 0.0
        mask = repeat(mask, "b n -> b d n", d=d)
        return (loss * mask).sum() / mask.sum()
