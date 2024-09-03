from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F  # noqa: N812
from torchdiffeq import odeint
from tqdm.auto import tqdm

from osu_fusion.library.osu.data.encode import TOTAL_DIM
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
        attn_heads: int = 16,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
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
            attn_context_len=attn_context_len,
        )

        self.sample_timesteps = sampling_timesteps
        self.cond_drop_prob = cond_drop_prob

    def set_full_bf16(self: "OsuFusion") -> None:
        self.unet = self.unet.bfloat16()

    @torch.inference_mode()
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
        with tqdm(total=(self.sample_timesteps - 1) * 2, desc="sampling loop time step", dynamic_ncols=True) as pbar:

            def ode_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                t_batched = repeat(t, "... -> b ...", b=b)
                out = self.unet.forward_with_cond_scale(x, a, t_batched, c, cond_scale=cond_scale)
                pbar.update(1)
                return out

            trajectory = odeint(ode_fn, x, times, method="midpoint", rtol=1e-5, atol=1e-5)
        return trajectory[-1]

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
        padded_times = rearrange(times, "b -> b () ()")

        t = cosmap(padded_times)
        x_noisy = t * x + (1 - t) * noise

        flow = x - noise
        pred = self.unet(x_noisy, a, times, c, cond_drop_prob=self.cond_drop_prob)

        # Calculate loss
        loss = F.mse_loss(pred, flow, reduction="none")

        # Create mask for losses to ignore padding
        if orig_len is not None:
            b, _, n = x.shape
            mask = torch.ones((b, n), device=x.device)
            for i, orig in enumerate(orig_len):
                mask[i, orig:] = 0.0
            mask = repeat(mask, "b n -> b d n", d=TOTAL_DIM)
            return (loss * mask).sum() / mask.sum()
        return loss.mean()
