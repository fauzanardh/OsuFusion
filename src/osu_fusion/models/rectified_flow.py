from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F
from torchdiffeq import odeint
from tqdm.auto import tqdm

from osu_fusion.data.const import AUDIO_DIM, BEATMAP_DIM, CONTEXT_DIM
from osu_fusion.models.backbone.unet import UNet


def cosmap(t: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 / (torch.tan(torch.pi / 2 * t) + 1))


class OsuFusion(nn.Module):
    def __init__(
        self: "OsuFusion",
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        num_layer_blocks: Tuple[int] = (3, 3, 3, 3),
        num_middle_transformers: int = 3,
        attn_dim_head: int = 64,
        attn_heads: int = 16,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
        cond_drop_prob: float = 0.5,
        sampling_timesteps: int = 16,
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

        self.sample_timesteps = sampling_timesteps
        self.cond_drop_prob = cond_drop_prob

    @property
    def trainable_params(self: "OsuFusion") -> Tuple[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]

    def set_full_bf16(self: "OsuFusion") -> None:
        self.unet = self.unet.bfloat16()

    @torch.inference_mode()
    def sample(
        self: "OsuFusion",
        n: int,
        a_lat: torch.Tensor,
        c_prep: torch.Tensor,
        c_prep_uncond: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        cond_scale: float = 2.0,
    ) -> torch.Tensor:
        assert cond_scale == 1.0 or c_prep_uncond is not None, "If cond_scale is not 1.0, c_uncond can't be None"

        b, device = a_lat.shape[0], a_lat.device
        if x is None:
            x = torch.randn((b, BEATMAP_DIM, n), device=device)

        times = torch.linspace(0.0, 1.0, self.sample_timesteps, device=device)
        with tqdm(total=(self.sample_timesteps - 1) * 2, desc="sampling loop time step", dynamic_ncols=True) as pbar:

            def ode_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                t_batched = repeat(t, "... -> b ...", b=b)
                out = self.unet.forward_with_cond_scale(
                    x,
                    a_lat,
                    t_batched,
                    c_prep,
                    c_prep_uncond,
                    cond_scale=cond_scale,
                )
                pbar.update(1)
                return out

            trajectory = odeint(ode_fn, x, times, method="midpoint", rtol=1e-5, atol=1e-5)
        return trajectory[-1]

    def forward(
        self: "OsuFusion",
        x: torch.Tensor,
        a_lat: torch.Tensor,
        c_prep: torch.Tensor,
        orig_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(x, device=x.device)
        timestep = torch.rand(x.shape[0], device=x.device)
        padded_timestep = rearrange(timestep, "b -> b () ()")

        t = cosmap(padded_timestep)
        x_noisy = t * x + (1 - t) * noise

        flow = x - noise
        pred = self.unet(x_noisy, a_lat, timestep, c_prep)

        # Calculate loss
        loss = F.mse_loss(pred, flow, reduction="none")
        return loss.mean()

        # # Create mask for losses to ignore padding
        # if orig_len is not None:
        #     b, _, n = x.shape
        #     mask = torch.ones((b, n), device=x.device)
        #     for i, orig in enumerate(orig_len):
        #         mask[i, orig:] = 0.0
        #     mask = repeat(mask, "b n -> b d n", d=BEATMAP_DIM)
        #     return (loss * mask).sum() / mask.sum()
        # return loss.mean()
