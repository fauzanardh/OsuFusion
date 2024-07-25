from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import repeat
from torch.nn import functional as F  # noqa: N812
from torchdiffeq import odeint
from tqdm.auto import tqdm

from osu_fusion.library.osu.data.encode import HIT_DIM, TOTAL_DIM
from osu_fusion.modules.unet import UNet
from osu_fusion.scripts.dataset_creator import AUDIO_DIM, CONTEXT_DIM


def cosmap(t: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 / (torch.tan(torch.pi / 2 * t) + 1))


class OsuFusion(nn.Module):
    def __init__(
        self: "OsuFusion",
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        num_layer_blocks: Tuple[int] = (3, 3, 3, 3),
        num_middle_transformers: int = 3,
        cross_embed_kernel_sizes: Tuple[int] = (3, 7, 15),
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 2,
        attn_qk_norm: bool = True,
        attn_causal: bool = False,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 8192,
        attn_infini: bool = True,
        attn_segment_len: int = 1024,
        cond_drop_prob: float = 0.5,
        sampling_timesteps: int = 16,
    ) -> None:
        super().__init__()

        self.unet = UNet(
            dim_in_x=TOTAL_DIM,
            dim_in_a=AUDIO_DIM,
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
        )

        self.sample_timesteps = sampling_timesteps
        self.cond_drop_prob = cond_drop_prob

    def set_full_bf16(self: "OsuFusion") -> None:
        self.unet = self.unet.bfloat16()

    def discretize_hit_features(self: "OsuFusion", x: torch.Tensor) -> torch.Tensor:
        hit_signals = x[:, :HIT_DIM, :]
        cursor_signals = x[:, HIT_DIM:, :]
        hit_signals = (hit_signals > 0.0).to(x.dtype) * 2 - 1
        return torch.cat([hit_signals, cursor_signals], dim=1)

    @torch.no_grad()
    def sample(
        self: "OsuFusion",
        a: torch.Tensor,
        c: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        cond_scale: float = 2.0,
    ) -> torch.Tensor:
        (b, _, n), device = a.shape, a.device
        if x is None:
            x = torch.randn((b, TOTAL_DIM, n), device=device)

        times = torch.linspace(0.0, 1.0, self.sample_timesteps, device=device)
        with tqdm(desc="sampling loop time step", dynamic_ncols=True) as pbar:

            def ode_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                t_batched = repeat(t, "... -> b ...", b=b)
                out = self.unet.forward_with_cond_scale(x, a, t_batched, c, cond_scale=cond_scale)
                pbar.update(1)
                return out

            trajectory = odeint(ode_fn, x, times, method="midpoint", rtol=1e-5, atol=1e-5)
        return self.discretize_hit_features(trajectory[-1])

    def forward(
        self: "OsuFusion",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
        orig_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert x.shape[-1] == a.shape[-1], "x and a must have the same number of sequence length"

        noise = torch.randn_like(x, device=x.device)
        times = torch.rand(x.shape[0], device=x.device)

        t = cosmap(times)
        x_noisy = t * x + (1 - t) * noise

        flow = x - noise
        pred = self.unet(x_noisy, a, t, c, cond_drop_prob=self.cond_drop_prob)

        # Calculate loss
        loss = F.mse_loss(pred, flow, reduction="none")

        # Create mask for losses to ignore padding
        b, _, n = x.shape
        mask = torch.ones((b, n), device=x.device)
        if orig_len is not None:
            for i, orig in enumerate(orig_len):
                mask[i, orig:] = 0.0
        mask = repeat(mask, "b n -> b d n", d=TOTAL_DIM)

        # Using mean instead of sum because if the sequence length is big
        # the intermediate loss values can be very big
        return (loss * mask).mean() / mask.mean()
