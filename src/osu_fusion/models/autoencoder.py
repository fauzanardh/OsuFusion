import itertools
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from osu_fusion.data.const import AUDIO_DIM, BEATMAP_DIM, HIT_DIM
from osu_fusion.modules.residual import ResidualBlock
from osu_fusion.modules.samplers import Downsample, Upsample
from osu_fusion.modules.transformer import TransformerBlock

DEBUG = os.environ.get("DEBUG", False)


def loss_fn_osu(
    x: torch.Tensor,
    recon_hit: torch.Tensor,
    recon_cursor: torch.Tensor,
    original_len: int,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    # Remove padding
    x = x[:, :, :original_len]
    hit_signals = x[:, :HIT_DIM]
    cursor_signals = x[:, HIT_DIM:]

    # Remove padding
    recon_hit = recon_hit[:, :, :original_len]
    recon_cursor = recon_cursor[:, :, :original_len]

    # Calculate hit signals using BCE loss and cursor signals using MSE loss
    hit_loss = F.binary_cross_entropy_with_logits(recon_hit, hit_signals)
    cursor_loss = F.mse_loss(recon_cursor, cursor_signals)

    # Calculate KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    return hit_loss, cursor_loss, kl_loss


class Block(nn.Module):
    def __init__(
        self: "Block",
        dim_in: int,
        dim_out: int,
        layer_idx: int,
        num_layers: int,
        down_block: bool = False,
        ff_mult: int = 2,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.init_resnet = ResidualBlock(dim_in, dim_out)
        self.resnet = ResidualBlock(dim_out, dim_out)
        self.transformer = TransformerBlock(
            dim_out,
            ff_mult=ff_mult,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len,
        )

        if down_block:
            self.sampler = (
                Downsample(dim_out, dim_out) if layer_idx < num_layers - 1 else nn.Conv1d(dim_out, dim_out, 1)
            )
        else:
            self.sampler = Upsample(dim_out, dim_out) if layer_idx < num_layers - 1 else nn.Conv1d(dim_out, dim_out, 1)

    def forward(self: "Block", x: torch.Tensor) -> torch.Tensor:
        x = self.init_resnet(x)
        x = self.resnet(x)
        x = self.transformer(x)

        return self.sampler(x)


class Encoder(nn.Module):
    def __init__(
        self: "Encoder",
        dim_in: int,
        dim_z: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        ff_mult: int = 2,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()

        self.init_conv = nn.Conv1d(dim_in, dim_h, 7, padding=3)

        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(itertools.pairwise(dims_h))
        n_layers = len(in_out)

        # Down
        down_blocks = []
        for i in range(n_layers):
            layer_dim_in, layer_dim_out = in_out[i]
            attn_context_len_layer = attn_context_len // (2**i)
            down_blocks.append(
                Block(
                    layer_dim_in,
                    layer_dim_out,
                    i,
                    n_layers,
                    down_block=True,
                    ff_mult=ff_mult,
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_kv_heads=attn_kv_heads,
                    attn_context_len=attn_context_len_layer,
                ),
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        # Middle
        self.middle_resnet = ResidualBlock(dims_h[-1], dims_h[-1])
        self.middle_attention = TransformerBlock(
            dims_h[-1],
            ff_mult=ff_mult,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len // (2 ** (n_layers - 1)),
        )
        self.middle_resnet2 = ResidualBlock(dims_h[-1], dims_h[-1])

        # End
        self.norm_out = nn.GroupNorm(1, dims_h[-1])
        self.to_z = nn.Conv1d(dims_h[-1], dim_z * 2, 7, padding=3)

    def forward(self: "Encoder", x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)

        # Down
        for block in self.down_blocks:
            x = block(x)

        # Middle
        x = self.middle_resnet(x)
        x = self.middle_attention(x)
        x = self.middle_resnet2(x)

        # End
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.to_z(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self: "Decoder",
        dim_out: int,
        dim_z: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        ff_mult: int = 2,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()

        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(reversed(tuple(itertools.pairwise(dims_h))))
        n_layers = len(in_out)

        self.init_conv = nn.Conv1d(dim_z, dims_h[-1], 7, padding=3)

        # Middle
        self.middle_resnet = ResidualBlock(dims_h[-1], dims_h[-1])
        self.middle_attention = TransformerBlock(
            dims_h[-1],
            ff_mult=ff_mult,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len // (2 ** (n_layers - 1)),
        )
        self.middle_resnet2 = ResidualBlock(dims_h[-1], dims_h[-1])

        # Up
        up_blocks = []
        for i in range(n_layers):
            layer_dim_out, layer_dim_in = in_out[i]
            attn_context_len_layer = attn_context_len // (2 ** (n_layers - i - 1))
            up_blocks.append(
                Block(
                    layer_dim_in,
                    layer_dim_out,
                    i,
                    n_layers,
                    down_block=False,
                    ff_mult=ff_mult,
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_kv_heads=attn_kv_heads,
                    attn_context_len=attn_context_len_layer,
                ),
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        # End
        self.norm_x = nn.GroupNorm(1, dims_h[0])
        self.to_out = nn.Conv1d(dims_h[0], dim_out, 7, padding=3)

    def forward(self: "Decoder", z: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(z)

        # Middle
        x = self.middle_resnet(x)
        x = self.middle_attention(x)
        x = self.middle_resnet2(x)

        # Up
        for block in self.up_blocks:
            x = block(x)

        # End
        x = self.norm_x(x)
        x = F.silu(x)
        return self.to_out(x)


class OsuAutoEncoder(nn.Module):
    def __init__(
        self: "OsuAutoEncoder",
        dim_z: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        ff_mult: int = 2,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        # Encoder
        self.encoder = Encoder(
            BEATMAP_DIM,
            dim_z,
            dim_h,
            dim_h_mult=dim_h_mult,
            ff_mult=ff_mult,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len,
        )

        # Decoder
        self.decoder = Decoder(
            BEATMAP_DIM,
            dim_z,
            dim_h,
            dim_h_mult=dim_h_mult,
            ff_mult=ff_mult,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len,
        )

    def encode(self: "OsuAutoEncoder", x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar

    def reparametrize(self: "OsuAutoEncoder", mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self: "OsuAutoEncoder", z: torch.Tensor, apply_act: bool = False) -> torch.Tensor:
        recon = self.decoder(z)
        recon_hit = torch.sigmoid(recon[:, :HIT_DIM]) if apply_act else recon[:, :HIT_DIM]
        recon_cursor = recon[:, HIT_DIM:]

        return recon_hit, recon_cursor

    def forward(self: "OsuAutoEncoder", x: torch.Tensor) -> torch.Tensor:
        # Pad to nearest power of 2^encoder layers
        n = x.shape[-1]
        depth = len(self.encoder.down_blocks)
        pad_len = (2**depth - (n % (2**depth))) % (2**depth)

        # pad hit signals with 0.0 and cursor signals with -1.0
        hit_signals = x[:, :HIT_DIM]
        cursor_signals = x[:, HIT_DIM:]
        hit_signals = F.pad(hit_signals, (0, pad_len), value=0.0)
        cursor_signals = F.pad(cursor_signals, (0, pad_len), value=-1.0)
        x = torch.cat([hit_signals, cursor_signals], dim=1)

        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon_hit, recon_cursor = self.decode(z)

        hit_loss, cursor_loss, kl_loss = loss_fn_osu(x, recon_hit, recon_cursor, n, mu, logvar)

        return hit_loss, cursor_loss, kl_loss


class AudioAutoEncoder(nn.Module):
    def __init__(
        self: "AudioAutoEncoder",
        dim_z: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        ff_mult: int = 2,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        # Encoder
        self.encoder = Encoder(
            AUDIO_DIM,
            dim_z,
            dim_h,
            dim_h_mult=dim_h_mult,
            ff_mult=ff_mult,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len,
        )

        # Decoder
        self.decoder = Decoder(
            AUDIO_DIM,
            dim_z,
            dim_h,
            dim_h_mult=dim_h_mult,
            ff_mult=ff_mult,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len,
        )

    def encode(self: "AudioAutoEncoder", x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar

    def reparametrize(self: "AudioAutoEncoder", mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self: "AudioAutoEncoder", z: torch.Tensor) -> torch.Tensor:
        audio = self.decoder(z)
        return audio

    def forward(self: "AudioAutoEncoder", x: torch.Tensor) -> torch.Tensor:
        # Pad to nearest power of 2^encoder layers
        n = x.shape[-1]
        depth = len(self.encoder.down_blocks)
        pad_len = (2**depth - (n % (2**depth))) % (2**depth)
        x = F.pad(x, (0, pad_len), value=0.0)

        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon_audio = self.decode(z)

        audio_loss = F.mse_loss(recon_audio[:, :, :n], x[:, :, :n])
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        return audio_loss, kl_loss
