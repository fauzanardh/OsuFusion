from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from einops import repeat
from torch.nn import functional as F
from tqdm.auto import tqdm

from osu_fusion.data.const import AUDIO_DIM, BEATMAP_LATENT_DIM, CONTEXT_DIM, HIT_DIM
from osu_fusion.models.autoencoder import AudioEncoder, OsuAutoEncoder
from osu_fusion.models.backbone.unet import UNet


class OsuFusion(nn.Module):
    def __init__(
        self: "OsuFusion",
        dim_h: int,
        dim_h_a: int,
        osu_autoencoder: OsuAutoEncoder,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        dim_h_a_mult: Tuple[int] = (1, 2, 4, 8),
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

        self.audio_encoder = AudioEncoder(
            dim_in=AUDIO_DIM,
            dim_h=dim_h_a,
            dim_h_mult=dim_h_a_mult,
            attn_context_len=attn_context_len,
        )
        self.unet = UNet(
            dim_in_x=BEATMAP_LATENT_DIM,
            dim_in_a=dim_h_a,
            dim_in_c=CONTEXT_DIM,
            dim_h=dim_h,
            dim_h_mult=dim_h_mult,
            num_layer_blocks=num_layer_blocks,
            num_middle_transformers=num_middle_transformers,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len
            // (2 ** (len(dim_h_mult) - 1)),  # Divide by autoencoder downsampling factor
        )

        self.osu_autoencoder = osu_autoencoder
        self.osu_autoencoder.eval()
        self.osu_autoencoder.requires_grad_(False)

        self.scheduler = DDIMScheduler(
            num_train_timesteps=train_timesteps,
            beta_schedule="linear",
        )
        self.train_timesteps = train_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.cond_drop_prob = cond_drop_prob

    @property
    def trainable_params(self: "OsuFusion") -> Tuple[nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)

    def set_full_bf16(self: "OsuFusion") -> None:
        self.unet = self.unet.bfloat16()

    @torch.no_grad()
    def encode_beatmap(self: "OsuFusion", x: torch.Tensor) -> torch.Tensor:
        # pad hit signals with 0.0 and cursor signals with -1.0
        n = x.shape[-1]
        depth = len(self.osu_autoencoder.encoder.down_blocks)
        pad_len = (2**depth - (n % (2**depth))) % (2**depth)
        hit_signals = x[:, :HIT_DIM]
        cursor_signals = x[:, HIT_DIM:]
        hit_signals = F.pad(hit_signals, (0, pad_len), value=0.0)
        cursor_signals = F.pad(cursor_signals, (0, pad_len), value=-1.0)
        x = torch.cat([hit_signals, cursor_signals], dim=1)

        z, _ = self.osu_autoencoder.encode(x)
        return z

    @torch.no_grad()
    def decode_beatmap_latent(self: "OsuFusion", z: torch.Tensor) -> torch.Tensor:
        recon_hit, recon_cursor = self.osu_autoencoder.decode(z, apply_act=True)
        x_recon = torch.cat([recon_hit, recon_cursor], dim=1)
        return x_recon

    def encode_audio(self: "OsuFusion", a: torch.Tensor) -> torch.Tensor:
        # pad audio signals with -23.0
        n = a.shape[-1]
        depth = len(self.audio_encoder.down_blocks)
        pad_len = (2**depth - (n % (2**depth))) % (2**depth)
        a = F.pad(a, (0, pad_len), value=-23.0)

        return self.audio_encoder(a)

    @torch.inference_mode()
    def sample(
        self: "OsuFusion",
        a: torch.Tensor,
        c: torch.Tensor,
        c_uncond: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        cond_scale: float = 7.0,
    ) -> torch.Tensor:
        assert cond_scale == 1.0 or c_uncond is not None, "If cond_scale is not 1.0, c_uncond can't be None"

        (b, _, n), device = a.shape, a.device
        if x is None:
            x = torch.randn((b, BEATMAP_LATENT_DIM, n), device=device)

        self.scheduler.set_timesteps(self.sampling_timesteps)
        for t in tqdm(self.scheduler.timesteps, desc="sampling loop time step", dynamic_ncols=True):
            t_batched = repeat(t, "... -> b ...", b=b).long().to(device)
            pred = self.unet.forward_with_cond_scale(x, a, t_batched, c, c_uncond, cond_scale=cond_scale)
            x = self.scheduler.step(pred, t, x).prev_sample

        return x

    def forward(
        self: "OsuFusion",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        assert x.shape[-1] == a.shape[-1], "x and a must have the same number of sequence length"

        noise = torch.randn_like(x, device=x.device)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (x.shape[0],),
            dtype=torch.int64,
            device=x.device,
        )
        x_noisy = self.scheduler.add_noise(x, noise, timesteps)

        pred = self.unet(x_noisy, a, timesteps, c)

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
