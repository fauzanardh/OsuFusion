import itertools
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from osu_fusion.modules.attention import Attention
from osu_fusion.modules.positional_embeddings import SinusoidalPositionEmbedding
from osu_fusion.modules.residual import ResidualBlock
from osu_fusion.modules.samplers import Downsample, Upsample
from osu_fusion.modules.transformer import TransformerBlock
from osu_fusion.modules.utils import prob_mask_like

DEBUG = os.environ.get("DEBUG", False)


def zero_init(module: nn.Module) -> nn.Module:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)

    return module


class UNetBlock(nn.Module):
    def __init__(
        self: "UNetBlock",
        dim_in: int,
        dim_out: int,
        dim_time: Optional[int],
        dim_cond: Optional[int],
        layer_idx: int,
        num_layers: int,
        num_blocks: int,
        down_block: bool,
        attn_dim_head: int,
        attn_heads: int,
        attn_kv_heads: int,
        attn_context_len: int,
    ) -> None:
        super().__init__()
        self.init_resnet = ResidualBlock(dim_in if down_block else dim_in + dim_out, dim_in, dim_time, dim_cond)
        self.resnets = nn.ModuleList(
            [ResidualBlock(dim_in, dim_in, dim_time, dim_cond) for _ in range(num_blocks)],
        )
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(
                    dim_in,
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_kv_heads=attn_kv_heads,
                    attn_context_len=attn_context_len,
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

        self.gradient_checkpointing = False

    def forward_body(
        self: "UNetBlock",
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.init_resnet(x, t, c)

        for resnet, transformer in zip(self.resnets, self.transformers, strict=True):
            x = resnet(x, t, c)
            x = transformer(x)

        return self.sampler(x), x

    def forward(
        self: "UNetBlock",
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, t, c, use_reentrant=True)
        else:
            return self.forward_body(x, t, c)


class AudioEncoder(nn.Module):
    # Simple audio encoder for now with resnet, plain attention, and downsampling
    def __init__(
        self: "AudioEncoder",
        dim_in: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        attn_dim_head: int = 64,
        attn_heads: int = 16,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h

        self.init_conv = nn.Conv1d(dim_in, dim_h, 7, padding=3)

        # Downsample
        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(itertools.pairwise(dims_h))
        n_layers = len(in_out)

        down_layers = []
        for i in range(n_layers):
            layer_dim_in, layer_dim_out = in_out[i]
            attn_context_len_layer = attn_context_len // (2**i)
            down_layers.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            layer_dim_in,
                            layer_dim_out,
                        ),
                        Attention(
                            layer_dim_out,
                            attn_dim_head,
                            attn_heads,
                            attn_kv_heads,
                            attn_context_len_layer,
                        ),
                        Downsample(layer_dim_out, layer_dim_out)
                        if i < (n_layers - 1)
                        else nn.Conv1d(layer_dim_out, layer_dim_out, 3, padding=1),
                    ],
                ),
            )
        self.down_layers = nn.ModuleList(down_layers)

        # Middle
        self.middle_resnet1 = ResidualBlock(dims_h[-1], dims_h[-1])
        self.middle_attention = Attention(
            dims_h[-1],
            attn_dim_head,
            attn_heads,
            attn_kv_heads,
            attn_context_len // (2 ** (n_layers - 1)),
        )
        self.middle_resnet2 = ResidualBlock(dims_h[-1], dims_h[-1])

    def forward(self: "AudioEncoder", a: torch.Tensor) -> torch.Tensor:
        a = self.init_conv(a)

        for resnet, attn, down in self.down_layers:
            a = resnet(a)
            a = rearrange(a, "b d n -> b n d")
            a = a + attn(a)
            a = rearrange(a, "b n d -> b d n")
            a = down(a)

        a = self.middle_resnet1(a)
        a = rearrange(a, "b d n -> b n d")
        a = a + self.middle_attention(a)
        a = rearrange(a, "b n d -> b d n")
        a = self.middle_resnet2(a)

        return a


class UNet(nn.Module):
    def __init__(
        self: "UNet",
        dim_in_x: int,
        dim_in_a: int,
        dim_in_c: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 4),
        dim_t: int = 256,
        num_layer_blocks: Tuple[int] = (3, 3, 3, 3),
        num_middle_transformers: int = 3,
        attn_dim_head: int = 64,
        attn_heads: int = 16,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.dim_emb = dim_h * 4

        self.init_x = nn.Conv1d(dim_in_x, dim_h, 7, padding=3)
        self.audio_encoder = AudioEncoder(
            dim_in=dim_in_a,
            dim_h=dim_h,
            dim_h_mult=dim_h_mult,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_context_len=attn_context_len,
        )

        self.final_resnet = ResidualBlock(dim_h * 2, dim_h, self.dim_emb, self.dim_emb)
        self.final_conv = zero_init(nn.Conv1d(dim_h, dim_in_x, 1))

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(dim_t),
            nn.Linear(dim_t, self.dim_emb),
            nn.SiLU(),
            nn.Linear(self.dim_emb, self.dim_emb),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(dim_in_c, self.dim_emb),
            nn.SiLU(),
            nn.Linear(self.dim_emb, self.dim_emb),
        )
        self.null_cond = nn.Parameter(torch.randn(self.dim_emb))

        # Downsample
        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(itertools.pairwise(dims_h))
        n_layers = len(in_out)

        down_layers = []
        for i in range(n_layers):
            layer_dim_in, layer_dim_out = in_out[i]
            num_blocks = num_layer_blocks[i]
            attn_context_len_layer = attn_context_len // (2**i)
            down_layers.append(
                UNetBlock(
                    layer_dim_in,
                    layer_dim_out,
                    self.dim_emb,
                    self.dim_emb,
                    i,
                    n_layers,
                    num_blocks,
                    True,
                    attn_dim_head,
                    attn_heads,
                    attn_kv_heads,
                    attn_context_len_layer,
                ),
            )
        self.down_layers = nn.ModuleList(down_layers)

        # Middle
        self.middle_resnet1 = ResidualBlock(
            dims_h[-1] * 2,
            dims_h[-1],
            self.dim_emb,
            self.dim_emb,
        )
        self.middle_transformer = nn.ModuleList(
            [
                TransformerBlock(
                    dims_h[-1],
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_kv_heads=attn_kv_heads,
                    attn_context_len=attn_context_len // (2 ** (n_layers - 1)),
                )
                for _ in range(num_middle_transformers)
            ],
        )
        self.middle_resnet2 = ResidualBlock(
            dims_h[-1],
            dims_h[-1],
            self.dim_emb,
            self.dim_emb,
        )

        # Upsample
        in_out = tuple(reversed(tuple(itertools.pairwise(dims_h))))
        num_layer_blocks = tuple(reversed(num_layer_blocks))
        n_layers = len(in_out)

        up_layers = []
        for i in range(n_layers):
            layer_dim_out, layer_dim_in = in_out[i]
            num_blocks = num_layer_blocks[i]
            attn_context_len_layer = attn_context_len // (2 ** (n_layers - i - 1))
            up_layers.append(
                UNetBlock(
                    layer_dim_in,
                    layer_dim_out,
                    self.dim_emb,
                    self.dim_emb,
                    i,
                    n_layers,
                    num_blocks,
                    False,
                    attn_dim_head,
                    attn_heads,
                    attn_kv_heads,
                    attn_context_len_layer,
                ),
            )
        self.up_layers = nn.ModuleList(up_layers)

    def set_gradient_checkpointing(self: "UNet", value: bool) -> None:
        for name, module in self.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
                print(f"Set gradient checkpointing to {value} for {name}")

    def encode_audio(self: "UNet", a: torch.Tensor) -> torch.Tensor:
        n = a.shape[-1]
        depth = len(self.down_layers)
        pad_len = (2**depth - (n % (2**depth))) % (2**depth)
        a = F.pad(a, (0, pad_len), value=0.0)
        return self.audio_encoder(a)

    def prepare_condition(self: "UNet", c: torch.Tensor, cond_drop_prob: float = 0.0) -> torch.Tensor:
        cond_mask = prob_mask_like((c.shape[0],), 1.0 - cond_drop_prob, device=c.device)
        cond_mask = rearrange(cond_mask, "b -> b 1")
        null_conds = repeat(self.null_cond, "d -> b d", b=c.shape[0])
        c = self.cond_mlp(c)
        return torch.where(cond_mask, c, null_conds)

    def forward_with_cond_scale(
        self: "UNet",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        c_uncond: torch.Tensor,
        cond_scale: float = 1.0,
    ) -> torch.Tensor:
        logits = self.forward(x, a, t, c)

        if cond_scale == 1.0:
            return logits

        null_logits = self.forward(x, a, t, c_uncond)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self: "UNet",
        x: torch.Tensor,
        a_lat: torch.Tensor,
        t: torch.Tensor,
        c_prep: torch.Tensor,
    ) -> torch.Tensor:
        n = x.shape[-1]
        depth = len(self.down_layers)
        pad_len = (2**depth - (n % (2**depth))) % (2**depth)
        x = F.pad(x, (0, pad_len), value=0.0)

        t = self.time_mlp(t)
        x = self.init_x(x)
        r = x.clone()

        skip_connection = []
        for down_layer in self.down_layers:
            x, skip = down_layer(x, t, c_prep)
            skip_connection.append(skip)

        x = torch.cat([x, a_lat], dim=1)
        x = self.middle_resnet1(x, t, c_prep)
        for transformer_block in self.middle_transformer:
            x = transformer_block(x)
        x = self.middle_resnet2(x, t, c_prep)

        for up_layer, skip in zip(self.up_layers, reversed(skip_connection), strict=True):
            x = torch.cat([x, skip], dim=1)
            x, _ = up_layer(x, t, c_prep)

        x = torch.cat([x, r], dim=1)
        x = self.final_resnet(x, t, c_prep)

        return self.final_conv(x)[:, :, :n]
