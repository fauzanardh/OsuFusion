import math
from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from osu_fusion.modules.residual import ResidualBlockV2
from osu_fusion.modules.transformer import Transformer
from osu_fusion.modules.utils import prob_mask_like


def zero_init_(module: nn.Module) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self: "LearnedSinusoidalPosEmb", dim: int) -> None:
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self: "LearnedSinusoidalPosEmb", x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        return fouriered


class UNet(nn.Module):
    def __init__(
        self: "UNet",
        dim_in: int,
        dim_out: int,
        dim_h: int,
        dim_cond: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        dim_learned_pos_emb: int = 16,
        num_time_tokens: int = 2,
        res_strides: Tuple[int] = (2, 2, 2, 2),
        res_num_layers: int = 4,
        attn_dim_head: int = 32,
        attn_heads: int = 8,
        attn_depth: int = 4,
        attn_dropout: float = 0.1,
        attn_sdpa: bool = True,
        attn_use_global_context_attention: bool = True,
        attn_use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.dim_emb = dim_h * 4

        self.pre_conv = nn.Conv1d(dim_in, dim_h, 7, padding=3)
        self.final_conv = nn.Conv1d(dim_h, dim_out, 1)
        zero_init_(self.final_conv)

        self.to_time_hiddens = nn.Sequential(
            LearnedSinusoidalPosEmb(dim_learned_pos_emb),
            nn.Linear(dim_learned_pos_emb, self.dim_emb),
            nn.SiLU(),
        )
        self.to_time_cond = nn.Linear(self.dim_emb, self.dim_emb)
        self.to_time_tokens = nn.Sequential(
            nn.Linear(self.dim_emb, dim_cond * num_time_tokens),
            Rearrange("b (r d) -> b r d", r=num_time_tokens),
        )
        self.null_cond = nn.Parameter(torch.randn(1, dim_cond))
        self.cond_norm = nn.LayerNorm(dim_cond)

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
            attn_use_rotary_emb=attn_use_rotary_emb,
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
                            use_rotary_emb=attn_use_rotary_emb,
                        ),
                        nn.Conv1d(layer_dim_in, layer_dim_out, 2 * stride, stride=stride, padding=1),
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
            use_rotary_emb=attn_use_rotary_emb,
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
                        nn.ConvTranspose1d(layer_dim_out, layer_dim_in, 2 * stride, stride=stride, padding=1),
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
                            use_rotary_emb=attn_use_rotary_emb,
                        ),
                    ],
                ),
            )
        self.up_layers = nn.ModuleList(up_layers)

        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self: "UNet", value: bool) -> None:
        self.gradient_checkpointing = value
        for name, module in self.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
                print(f"Set gradient checkpointing to {value} for {name}")

    def forward_with_cond_scale(self: "UNet", *args: List, cond_scale: float = 1.0, **kwargs: Dict) -> torch.Tensor:
        logits = self(*args, **kwargs)

        if cond_scale == 1.0:
            return logits

        null_logits = self(*args, **kwargs, cond_drop_prob=1.0)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self: "UNet",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        x = torch.cat([x, a], dim=1)
        x = self.pre_conv(x)

        time_hiddens = self.to_time_hiddens(t)
        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        c = rearrange(c, "b d -> b 1 d")
        cond_mask = prob_mask_like((x.shape[0],), 1.0 - cond_drop_prob, device=x.device)
        cond_mask = rearrange(cond_mask, "b -> b 1 1")
        c = torch.where(cond_mask, c, self.null_cond)

        c = torch.cat([time_tokens, c], dim=1)
        c = self.cond_norm(c)

        skip_connections = []
        for down_init, down_blocks, down_transformer, downsample in self.down_layers:
            x = down_init(x, t, c)
            for down_resnet in down_blocks:
                x = down_resnet(x, t)
                skip_connections.append(x)
            x = down_transformer(x, c)
            skip_connections.append(x)
            x = downsample(x)

        x = self.middle_resnet1(x, t)
        x = self.middle_transformer(x, c)
        x = self.middle_resnet2(x, t)

        for upsample, up_init, up_blocks, up_transformer in self.up_layers:
            x = upsample(x)
            x = up_init(torch.cat([x, skip_connections.pop()], dim=1), t, c)
            for up_resnet in up_blocks:
                x = up_resnet(torch.cat([x, skip_connections.pop()], dim=1), t)
            x = up_transformer(x, c)

        return self.final_conv(x)
