import os
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F  # noqa: N812
from torch.profiler import record_function

from osu_fusion.library.dataset import AUDIO_DIM
from osu_fusion.library.osu.data.encode import CURSOR_DIM, HIT_DIM
from osu_fusion.modules.attention import Attend, RotaryPositionEmbedding
from osu_fusion.modules.residual import ResidualBlock
from osu_fusion.modules.utils import dummy_context_manager

DEBUG = os.environ.get("DEBUG", False)


def loss_fn_osu(
    x: torch.Tensor,
    recon_hit: torch.Tensor,
    recon_cursor: torch.Tensor,
    original_len: int,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    dtype = recon_hit.dtype
    hit_signals = (x[:, :HIT_DIM] >= 0).to(dtype=dtype)
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


class Upsample(nn.Module):
    def __init__(self: "Upsample", dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, 3, padding=1)

    def forward_body(self: "Upsample", x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

    def forward(self: "Upsample", x: torch.Tensor) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("Upsample")
        with context_manager:
            return self.forward_body(x)


class Downsample(nn.Module):
    def __init__(self: "Downsample", dim_in: int, dim_out: int) -> None:
        super().__init__()
        # Asymmetric padding
        self.conv = nn.Conv1d(dim_in, dim_out, 3, stride=2, padding=0)

    def forward_body(self: "Downsample", x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1)
        x = F.pad(x, pad=pad, mode="reflect")
        x = self.conv(x)
        return x

    def forward(self: "Downsample", x: torch.Tensor) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("Downsample")
        with context_manager:
            return self.forward_body(x)


class Attention(nn.Module):
    def __init__(
        self: "Attention",
        dim_in: int,
        dim_head: int,
        heads: int,
        kv_heads: int,
        context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads

        self.norm = nn.LayerNorm(dim_in)
        self.to_q = nn.Linear(dim_in, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim_in, dim_head * kv_heads * 2, bias=False)
        self.rotary_emb = RotaryPositionEmbedding(dim_head, scale_base=context_len)

        self.attn = Attend()
        self.to_out = nn.Linear(dim_head * heads, dim_in)

    def forward_body(self: "Attention", x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b d n -> b n d")

        # Pre-norm
        x = self.norm(x)

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)

        k, v = self.to_kv(x).chunk(2, dim=-1)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.kv_heads) for t in (k, v))

        # GQA
        k, v = (repeat(t, "b h n d -> b (r h) n d", r=self.heads // self.kv_heads) for t in (k, v))

        q, k = self.rotary_emb(q, k)

        out = self.attn(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return rearrange(x + self.to_out(out), "b n d -> b d n")

    def forward(self: "Attention", x: torch.Tensor) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("Attention")
        with context_manager:
            return self.forward_body(x)


class Block(nn.Module):
    def __init__(
        self: "Block",
        dim_in: int,
        dim_out: int,
        layer_idx: int,
        num_layers: int,
        down_block: bool = False,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.init_resnet = ResidualBlock(dim_in, dim_out)
        self.resnet = ResidualBlock(dim_out, dim_out)
        self.attention = Attention(
            dim_out,
            attn_dim_head,
            attn_heads,
            attn_kv_heads,
            context_len=attn_context_len,
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
        x = self.attention(x)

        return self.sampler(x)


class Encoder(nn.Module):
    def __init__(
        self: "Encoder",
        dim_in: int,
        dim_z: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()

        self.init_conv = nn.Conv1d(dim_in, dim_h, 3, padding=1)

        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(zip(dims_h[:-1], dims_h[1:]))
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
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_kv_heads=attn_kv_heads,
                    attn_context_len=attn_context_len_layer,
                ),
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        # Middle
        self.middle_resnet = ResidualBlock(dims_h[-1], dims_h[-1])
        self.middle_attention = Attention(
            dims_h[-1],
            attn_dim_head,
            attn_heads,
            attn_kv_heads,
            context_len=attn_context_len,
        )
        self.middle_resnet2 = ResidualBlock(dims_h[-1], dims_h[-1])

        # End
        self.norm_out = nn.GroupNorm(1, dims_h[-1])
        self.to_z = nn.Conv1d(dims_h[-1], dim_z * 2, 3, padding=1)

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


class OsuDecoder(nn.Module):
    def __init__(
        self: "OsuDecoder",
        dim_out: int,
        dim_z: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()

        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(reversed(tuple(zip(dims_h[:-1], dims_h[1:]))))
        n_layers = len(in_out)

        self.init_conv = nn.Conv1d(dim_z, dims_h[-1], 3, padding=1)

        # Middle
        self.middle_resnet = ResidualBlock(dims_h[-1], dims_h[-1])
        self.middle_attention = Attention(
            dims_h[-1],
            attn_dim_head,
            attn_heads,
            attn_kv_heads,
            context_len=attn_context_len,
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
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_kv_heads=attn_kv_heads,
                    attn_context_len=attn_context_len_layer,
                ),
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        # End
        self.norm_x = nn.GroupNorm(1, dims_h[0])
        self.to_x = nn.Conv1d(dims_h[0], dim_out, 3, padding=1)

        # Try to output to the input rather than the embedding
        self.to_hit = nn.Linear(dim_out, HIT_DIM)
        self.to_cursor = nn.Linear(dim_out, CURSOR_DIM)

    def forward(self: "OsuDecoder", z: torch.Tensor) -> torch.Tensor:
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
        x = self.to_x(x)

        # Convert to hit and cursor signals
        x = rearrange(x, "b d n -> b n d")
        hit = self.to_hit(x)
        cursor = F.tanh(self.to_cursor(x))
        return hit, cursor


class AudioDecoder(nn.Module):
    def __init__(
        self: "AudioDecoder",
        dim_out: int,
        dim_z: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()

        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(reversed(tuple(zip(dims_h[:-1], dims_h[1:]))))
        n_layers = len(in_out)

        self.init_conv = nn.Conv1d(dim_z, dims_h[-1], 3, padding=1)

        # Middle
        self.middle_resnet = ResidualBlock(dims_h[-1], dims_h[-1])
        self.middle_attention = Attention(
            dims_h[-1],
            attn_dim_head,
            attn_heads,
            attn_kv_heads,
            context_len=attn_context_len,
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
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_kv_heads=attn_kv_heads,
                    attn_context_len=attn_context_len_layer,
                ),
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        # End
        self.norm_x = nn.GroupNorm(1, dims_h[0])
        self.to_x = nn.Conv1d(dims_h[0], dim_out, 3, padding=1)

        # Try to output to the input rather than the embedding
        self.to_audio = nn.Linear(dim_out, AUDIO_DIM)

    def forward(self: "AudioDecoder", z: torch.Tensor) -> torch.Tensor:
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
        x = self.to_x(x)

        # Convert to audio signals
        x = rearrange(x, "b d n -> b n d")
        audio = self.to_audio(x)
        return audio


class OsuAutoEncoder(nn.Module):
    def __init__(
        self: "OsuAutoEncoder",
        dim_emb: int,
        dim_z: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        assert dim_emb % HIT_DIM == 0, "dim_emb must be divisible by HIT_DIM"

        # Embeddings
        hit_emb_dim = dim_emb // HIT_DIM
        # Discrete hit signals embedding
        self.hit_embedding = nn.Embedding(2, hit_emb_dim)
        # Continuous cursor signals embedding
        self.cursor_embedding = nn.Sequential(
            nn.Linear(CURSOR_DIM, dim_emb),
            nn.SiLU(),
            nn.Linear(dim_emb, dim_emb),
        )
        self.mlp_embedding = nn.Sequential(
            nn.Linear(dim_emb * 2, dim_emb),
            nn.SiLU(),
            nn.Linear(dim_emb, dim_emb),
        )

        # Encoder
        self.encoder = Encoder(
            dim_emb,
            dim_z,
            dim_h,
            dim_h_mult=dim_h_mult,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len,
        )

        # Decoder
        self.decoder = OsuDecoder(
            dim_emb,
            dim_z,
            dim_h,
            dim_h_mult=dim_h_mult,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len,
        )

    def encode(self: "OsuAutoEncoder", x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = rearrange(x, "b d n -> b n d")
        # convert hit signals, 0 if x < 0, 1 if x >= 0
        hit_signals = (x[:, :, :HIT_DIM] >= 0).long()
        hit_emb = self.hit_embedding(hit_signals)
        hit_emb = rearrange(hit_emb, "b n d e -> b n (d e)")

        # convert cursor signals to embedding
        cursor_signals = x[:, :, HIT_DIM:]
        cursor_emb = self.cursor_embedding(cursor_signals)

        # concatenate hit and cursor embeddings
        emb = torch.cat([hit_emb, cursor_emb], dim=-1)
        emb = self.mlp_embedding(emb)
        emb = rearrange(emb, "b n d -> b d n")

        mu, logvar = self.encoder(emb).chunk(2, dim=1)
        return mu, logvar

    def reparametrize(self: "OsuAutoEncoder", mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self: "OsuAutoEncoder", z: torch.Tensor, use_tanh: bool = False) -> torch.Tensor:
        recon_hit, recon_cursor = self.decoder(z)
        recon_hit = rearrange(recon_hit, "b n d -> b d n")
        recon_cursor = rearrange(recon_cursor, "b n d -> b d n")

        if use_tanh:
            # Only apply tanh to hit signals since cursor signals are already in [-1, 1]
            recon_hit = F.tanh(recon_hit)  # Rescale to [-1, 1]
        return recon_hit, recon_cursor

    def forward(self: "OsuAutoEncoder", x: torch.Tensor) -> torch.Tensor:
        # Pad to nearest power of 2^encoder layers
        n = x.shape[-1]
        depth = len(self.encoder.down_blocks)
        pad_len = (2**depth - (n % (2**depth))) % (2**depth)
        x = F.pad(x, (0, pad_len), value=-1.0)

        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon_hit, recon_cursor = self.decode(z)

        hit_loss, cursor_loss, kl_loss = loss_fn_osu(x, recon_hit, recon_cursor, n, mu, logvar)

        return hit_loss, cursor_loss, kl_loss


class AudioAutoEncoder(nn.Module):
    def __init__(
        self: "AudioAutoEncoder",
        dim_emb: int,
        dim_z: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()

        # Embeddings
        self.embedding = nn.Sequential(
            nn.Linear(AUDIO_DIM, dim_emb),
            nn.SiLU(),
            nn.Linear(dim_emb, dim_emb),
        )

        # Encoder
        self.encoder = Encoder(
            dim_emb,
            dim_z,
            dim_h,
            dim_h_mult=dim_h_mult,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len,
        )

        # Decoder
        self.decoder = AudioDecoder(
            dim_emb,
            dim_z,
            dim_h,
            dim_h_mult=dim_h_mult,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len,
        )

    def encode(self: "AudioAutoEncoder", x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = rearrange(x, "b d n -> b n d")
        emb = self.embedding(x)
        emb = rearrange(emb, "b n d -> b d n")

        mu, logvar = self.encoder(emb).chunk(2, dim=1)
        return mu, logvar

    def reparametrize(self: "AudioAutoEncoder", mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self: "AudioAutoEncoder", z: torch.Tensor) -> torch.Tensor:
        audio = self.decoder(z)
        audio = rearrange(audio, "b n d -> b d n")
        return audio

    def forward(self: "AudioAutoEncoder", x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon_audio = self.decode(z)

        audio_loss = F.mse_loss(recon_audio, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        return audio_loss, kl_loss
