import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from osu_fusion.modules.causal_convolution import CausalConv1d, CausalConvTranspose1d
from osu_fusion.modules.residual import ResidualBlockV2
from osu_fusion.modules.transformer import Transformer


def zero_init_(module: nn.Module) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class LearnedSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self: "LearnedSinusoidalPositionalEmbedding", dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "Dim must be divisible by 2"
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self: "LearnedSinusoidalPositionalEmbedding", x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fourier = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        fourier = torch.cat([x, fourier], dim=-1)
        return fourier


class UNet(nn.Module):
    def __init__(
        self: "UNet",
        dim_in_x: int,
        dim_in_a: int,
        dim_out: int,
        dim_h: int,
        dim_cond: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        dim_learned_sinu: int = 16,
        res_strides: Tuple[int] = (2, 2, 2, 2),
        res_num_layers: int = 4,
        attn_dim_head: int = 32,
        attn_heads: int = 8,
        attn_depth: int = 4,
        attn_dropout: float = 0.1,
        attn_sdpa: bool = True,
        attn_use_global_context_attention: bool = True,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.dim_emb = dim_h * 4

        self.pre_conv = CausalConv1d(dim_in_x + dim_in_a, dim_h, 7)
        self.audio_pre_batchnorm = nn.BatchNorm1d(dim_in_a)
        self.final_conv = CausalConv1d(dim_h, dim_out, 1)
        zero_init_(self.final_conv.conv)

        self.time_embedding = nn.Sequential(
            LearnedSinusoidalPositionalEmbedding(dim_learned_sinu),
            nn.Linear(dim_learned_sinu + 1, self.dim_emb),
            nn.SiLU(),
            nn.Linear(self.dim_emb, self.dim_emb),
        )

        # Downsample
        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(zip(dims_h[:-1], dims_h[1:]))
        n_layers = len(in_out)

        resnet_block_attn = partial(
            ResidualBlockV2,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
            attn_sdpa=attn_sdpa,
        )

        skip_connection_dims = []
        down_layers = []
        for i in range(n_layers):
            layer_dim_in, layer_dim_out = in_out[i]
            stride = res_strides[i]
            down_layers.append(
                nn.ModuleList(
                    [
                        resnet_block_attn(
                            layer_dim_in,
                            layer_dim_in,
                            self.dim_emb,
                            dim_context=dim_cond,
                        ),
                        nn.ModuleList(
                            [
                                ResidualBlockV2(
                                    layer_dim_in,
                                    layer_dim_in,
                                    self.dim_emb,
                                    use_gca=attn_use_global_context_attention,
                                )
                                for _ in range(res_num_layers)
                            ],
                        ),
                        Transformer(
                            layer_dim_in,
                            dim_cond,
                            dim_head=attn_dim_head,
                            heads=attn_heads,
                            depth=attn_depth,
                            dropout=attn_dropout,
                            sdpa=attn_sdpa,
                        ),
                        CausalConv1d(layer_dim_in, layer_dim_out, 2 * stride, stride=stride),
                    ],
                ),
            )
            skip_connection_dims.append(layer_dim_in)
        self.down_layers = nn.ModuleList(down_layers)

        # Middle
        self.middle_resnet1 = ResidualBlockV2(
            dims_h[-1],
            dims_h[-1],
            self.dim_emb,
            use_gca=attn_use_global_context_attention,
        )
        self.middle_transformer = Transformer(
            dims_h[-1],
            dim_cond,
            dim_head=attn_dim_head,
            heads=attn_heads,
            depth=attn_depth,
            dropout=attn_dropout,
            sdpa=attn_sdpa,
        )
        self.middle_resnet2 = ResidualBlockV2(
            dims_h[-1],
            dims_h[-1],
            self.dim_emb,
            use_gca=attn_use_global_context_attention,
        )

        # Upsample
        res_strides = tuple(reversed(res_strides))
        in_out = tuple(reversed(tuple(zip(dims_h[:-1], dims_h[1:]))))
        n_layers = len(in_out)

        up_layers = []
        for i in range(n_layers):
            layer_dim_in, layer_dim_out = in_out[i]
            skip_connection_dim = skip_connection_dims.pop()
            stride = res_strides[i]
            up_layers.append(
                nn.ModuleList(
                    [
                        CausalConvTranspose1d(layer_dim_out, layer_dim_in, 2 * stride, stride=stride),
                        resnet_block_attn(
                            layer_dim_in + skip_connection_dim,
                            layer_dim_in,
                            self.dim_emb,
                            dim_context=dim_cond,
                        ),
                        nn.ModuleList(
                            [
                                ResidualBlockV2(
                                    layer_dim_in + skip_connection_dim,
                                    layer_dim_in,
                                    self.dim_emb,
                                    use_gca=attn_use_global_context_attention,
                                )
                                for _ in range(res_num_layers)
                            ],
                        ),
                        Transformer(
                            layer_dim_in,
                            dim_cond,
                            dim_head=attn_dim_head,
                            heads=attn_heads,
                            depth=attn_depth,
                            dropout=attn_dropout,
                            sdpa=attn_sdpa,
                        ),
                    ],
                ),
            )
        self.up_layers = nn.ModuleList(up_layers)

    def forward(self: "UNet", x: torch.Tensor, a: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        a = self.audio_pre_batchnorm(a)
        x = torch.cat([x, a], dim=1)
        x = self.pre_conv(x)

        t_emb = self.time_embedding(t)

        skip_connections = []
        for down_init, down_blocks, down_transformer, downsample in self.down_layers:
            x = down_init(x, t_emb, c)
            for down_resnet in down_blocks:
                x = down_resnet(x, t_emb)
                skip_connections.append(x)
            x = down_transformer(x, c)
            skip_connections.append(x)
            x = downsample(x)

        x = self.middle_resnet1(x, t_emb)
        x = self.middle_transformer(x, c)
        x = self.middle_resnet2(x, t_emb)

        for upsample, up_init, up_blocks, up_transformer in self.up_layers:
            x = upsample(x)
            x = up_init(torch.cat([x, skip_connections.pop()], dim=1), t_emb, c)
            for up_resnet in up_blocks:
                x = up_resnet(torch.cat([x, skip_connections.pop()], dim=1), t_emb)
            x = up_transformer(x, c)

        return self.final_conv(x)
