from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, repeat

from osu_fusion.modules.attention import Attend
from osu_fusion.modules.residual import ResidualBlock


class Upsample(nn.Sequential):
    def __init__(self: "Upsample", dim_in: int, dim_out: int) -> None:
        super().__init__(
            nn.ConvTranspose1d(dim_in, dim_out, 4, stride=2, padding=1),
        )


class Downsample(nn.Sequential):
    def __init__(self: "Downsample", dim_in: int, dim_out: int) -> None:
        super().__init__(
            nn.Conv1d(dim_in, dim_out, 4, stride=2, padding=1, padding_mode="reflect"),
        )


class MultiHeadRMSNorm(nn.Module):
    def __init__(self: "MultiHeadRMSNorm", dim: int, heads: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self: "MultiHeadRMSNorm", x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class Attention(nn.Module):
    def __init__(
        self: "Attention",
        dim_in: int,
        dim_head: int,
        heads: int,
        kv_heads: int,
        qk_norm: bool = True,
        causal: bool = False,
        use_rotary_emb: bool = True,
        context_len: int = 4096,
        infini: bool = False,
        segment_len: int = 1024,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.qk_norm = qk_norm

        self.to_q = nn.Linear(dim_in, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim_in, dim_head * kv_heads * 2, bias=False)
        self.q_norm = MultiHeadRMSNorm(dim_head, heads) if qk_norm else None
        self.k_norm = MultiHeadRMSNorm(dim_head, kv_heads) if qk_norm else None

        self.attn = Attend(
            dim_head,
            heads=heads,
            causal=causal,
            use_rotary_emb=use_rotary_emb,
            context_len=context_len,
            infini=infini,
            segment_len=segment_len,
        )
        self.linear = nn.Linear(dim_head * heads, dim_in)

    def forward(self: "Attention", x: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)

        k, v = self.to_kv(x).chunk(2, dim=-1)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.kv_heads) for t in (k, v))
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # GQA
        k, v = (repeat(t, "b h n d -> b (r h) n d", r=self.heads // self.kv_heads) for t in (k, v))

        out = self.attn(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return x + self.linear(out)


class FeedForward(nn.Sequential):
    def __init__(self: "FeedForward", dim: int, dim_mult: int = 2) -> None:
        inner_dim = dim * dim_mult
        super().__init__(
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, dim),
        )


class TransformerBlock(nn.Module):
    def __init__(
        self: "TransformerBlock",
        dim: int,
        ff_mult: int = 2,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 2,
        attn_qk_norm: bool = True,
        attn_causal: bool = False,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 8192,
        attn_infini: bool = True,
        attn_segment_len: int = 1024,
    ) -> None:
        super().__init__()
        self.attn = Attention(
            dim,
            attn_dim_head,
            attn_heads,
            attn_kv_heads,
            attn_qk_norm,
            attn_causal,
            attn_use_rotary_emb,
            attn_context_len,
            attn_infini,
            attn_segment_len,
        )
        self.ff = FeedForward(dim, ff_mult)

    def forward(self: "TransformerBlock", x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b d n -> b n d")
        x = self.attn(x)
        x = self.ff(x) + x
        return rearrange(x, "b n d -> b d n")


class Block(nn.Module):
    def __init__(
        self: "Block",
        dim_in: int,
        dim_out: int,
        layer_idx: int,
        num_layers: int,
        num_blocks: int,
        down_block: bool,
        attn_dim_head: int,
        attn_heads: int,
        attn_kv_heads: int,
        attn_qk_norm: bool,
        attn_causal: bool,
        attn_use_rotary_emb: bool,
        attn_context_len: int,
        attn_infini: bool,
        attn_segment_len: int,
    ) -> None:
        super().__init__()
        self.init_resnet = ResidualBlock(dim_in, dim_in)
        self.resnets = nn.ModuleList(
            [ResidualBlock(dim_in, dim_in) for _ in range(num_blocks)],
        )
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    dim_in,
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
                for _ in range(num_blocks)
            ],
        )
        if down_block:
            self.sampler = (
                Downsample(dim_in, dim_out)
                if layer_idx < (num_layers - 1)
                else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            )
        else:
            self.sampler = (
                Upsample(dim_in, dim_out) if layer_idx < (num_layers - 1) else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            )

    def forward(self: "Block", x: torch.Tensor) -> torch.Tensor:
        x = self.init_resnet(x)

        for resnet, transformer in zip(self.resnets, self.transformer):
            x = resnet(x)
            x = transformer(x)

        return self.sampler(x)


class Encoder(nn.Module):
    def __init__(
        self: "Encoder",
        dim_in: int,
        dim_z: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 4),
        num_layer_blocks: Tuple[int] = (2, 2, 2, 2),
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 2,
        attn_qk_norm: bool = True,
        attn_causal: bool = False,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 8192,
        attn_infini: bool = False,
        attn_segment_len: int = 1024,
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
            num_blocks = num_layer_blocks[i]
            attn_context_len_layer = attn_context_len // (2**i)
            attn_segment_len_layer = attn_segment_len // (2**i)
            down_blocks.append(
                Block(
                    layer_dim_in,
                    layer_dim_out,
                    i,
                    n_layers,
                    num_blocks,
                    True,
                    attn_dim_head,
                    attn_heads,
                    attn_kv_heads,
                    attn_qk_norm,
                    attn_causal,
                    attn_use_rotary_emb,
                    attn_context_len_layer,
                    attn_infini,
                    attn_segment_len_layer,
                ),
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        # Middle
        self.middle_resnet1 = ResidualBlock(dims_h[-1], dims_h[-1])
        self.middle_transformer = TransformerBlock(
            dims_h[-1],
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_qk_norm=attn_qk_norm,
            attn_causal=attn_causal,
            attn_use_rotary_emb=attn_use_rotary_emb,
            attn_context_len=attn_context_len // (2 ** (n_layers - 1)),
            attn_infini=attn_infini,
            attn_segment_len=attn_segment_len // (2 ** (n_layers - 1)),
        )
        self.middle_resnet2 = ResidualBlock(dims_h[-1], dims_h[-1])

        # End
        self.norm_out = nn.GroupNorm(1, dims_h[-1])
        self.conv_out = nn.Conv1d(dims_h[-1], 2 * dim_z, 3, padding=1)

    def forward(self: "Encoder", x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)

        # Down
        for block in self.down_blocks:
            x = block(x)

        # Middle
        x = self.middle_resnet1(x)
        x = self.middle_transformer(x)
        x = self.middle_resnet2(x)

        # End
        x = self.norm_out(x)
        x = x * torch.sigmoid(x)  # Swish
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self: "Decoder",
        dim_out: int,
        dim_z: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 4),
        num_layer_blocks: Tuple[int] = (2, 2, 2, 2),
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 2,
        attn_qk_norm: bool = True,
        attn_causal: bool = False,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 8192,
        attn_infini: bool = False,
        attn_segment_len: int = 1024,
    ) -> None:
        super().__init__()

        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(reversed(tuple(zip(dims_h[:-1], dims_h[1:]))))
        num_layer_blocks = tuple(reversed(num_layer_blocks))
        n_layers = len(in_out)

        self.init_conv = nn.Conv1d(dim_z, dims_h[-1], 3, padding=1)

        # Middle
        self.middle_resnet1 = ResidualBlock(dims_h[-1], dims_h[-1])
        self.middle_transformer = TransformerBlock(
            dims_h[-1],
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_qk_norm=attn_qk_norm,
            attn_causal=attn_causal,
            attn_use_rotary_emb=attn_use_rotary_emb,
            attn_context_len=attn_context_len // (2 ** (n_layers - 1)),
            attn_infini=attn_infini,
            attn_segment_len=attn_segment_len // (2 ** (n_layers - 1)),
        )
        self.middle_resnet2 = ResidualBlock(dims_h[-1], dims_h[-1])

        # Up
        up_blocks = []
        for i in range(n_layers):
            layer_dim_out, layer_dim_in = in_out[i]
            num_blocks = num_layer_blocks[i]
            attn_context_len_layer = attn_context_len // (2 ** (n_layers - i - 1))
            attn_segment_len_layer = attn_segment_len // (2 ** (n_layers - i - 1))
            up_blocks.append(
                Block(
                    layer_dim_in,
                    layer_dim_out,
                    i,
                    n_layers,
                    num_blocks,
                    False,
                    attn_dim_head,
                    attn_heads,
                    attn_kv_heads,
                    attn_qk_norm,
                    attn_causal,
                    attn_use_rotary_emb,
                    attn_context_len_layer,
                    attn_infini,
                    attn_segment_len_layer,
                ),
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        # End
        self.norm_out = nn.GroupNorm(1, dims_h[0])
        self.conv_out = nn.Conv1d(dims_h[0], dim_out, 3, padding=1)

    def forward(self: "Decoder", x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)

        # Middle
        x = self.middle_resnet1(x)
        x = self.middle_transformer(x)
        x = self.middle_resnet2(x)

        # Up
        for block in self.up_blocks:
            x = block(x)

        # End
        x = self.norm_out(x)
        x = x * torch.sigmoid(x)  # Swish
        x = self.conv_out(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(
        self: "AutoEncoder",
        dim_in: int,
        dim_z: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 4),
        num_layer_blocks: Tuple[int] = (2, 2, 2, 2),
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 2,
        attn_qk_norm: bool = True,
        attn_causal: bool = False,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 8192,
        attn_infini: bool = False,
        attn_segment_len: int = 1024,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            dim_in,
            dim_z,
            dim_h,
            dim_h_mult,
            num_layer_blocks,
            attn_dim_head,
            attn_heads,
            attn_kv_heads,
            attn_qk_norm,
            attn_causal,
            attn_use_rotary_emb,
            attn_context_len,
            attn_infini,
            attn_segment_len,
        )
        self.decoder = Decoder(
            dim_in,
            dim_z,
            dim_h,
            dim_h_mult,
            num_layer_blocks,
            attn_dim_head,
            attn_heads,
            attn_kv_heads,
            attn_qk_norm,
            attn_causal,
            attn_use_rotary_emb,
            attn_context_len,
            attn_infini,
            attn_segment_len,
        )

    def encode(self: "AutoEncoder", x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self: "AutoEncoder", mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self: "AutoEncoder", z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self: "AutoEncoder", x: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)

        recon_loss = F.mse_loss(reconstructed, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        return recon_loss, kl_loss
