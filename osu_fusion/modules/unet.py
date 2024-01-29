import math
from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat

from osu_fusion.modules.residual import ResidualBlock
from osu_fusion.modules.transformer import Transformer
from osu_fusion.modules.utils import prob_mask_like


def zero_init_(module: nn.Module) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class SinusoidalPosEmb(nn.Module):
    def __init__(self: "SinusoidalPosEmb", dim: int, theta: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self: "SinusoidalPosEmb", x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, "b -> b 1") * rearrange(emb, "d -> 1 d")
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


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


class UNet(nn.Module):
    def __init__(
        self: "UNet",
        dim_in: int,
        dim_out: int,
        dim_h: int,
        dim_cond: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        num_time_tokens: int = 2,
        resnet_depths: Tuple[int] = (2, 2, 2, 2),
        attn_dim_head: int = 32,
        attn_heads: int = 8,
        attn_depths: Tuple[int] = (4, 4, 4, 4),
        attn_dropout: float = 0.0,
        attn_sdpa: bool = True,
        attn_use_global_context_attention: bool = True,
        attn_use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.dim_emb = dim_h * 4
        self.dim_cond = dim_cond * 4

        resnet_block_attn = partial(
            ResidualBlock,
            dim_context=self.dim_cond,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
            attn_sdpa=attn_sdpa,
            attn_use_rotary_emb=attn_use_rotary_emb,
        )

        self.pre_conv = nn.Conv1d(dim_in, dim_h, 7, padding=3)
        self.final_conv = nn.Conv1d(dim_h, dim_out, 1)
        zero_init_(self.final_conv)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.dim_emb),
            nn.Linear(self.dim_emb, self.dim_emb),
            nn.SiLU(),
            nn.Linear(self.dim_emb, self.dim_emb),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(dim_cond, self.dim_cond),
            nn.SiLU(),
            nn.Linear(self.dim_cond, self.dim_cond),
        )
        self.null_cond = nn.Parameter(torch.randn(dim_cond))
        self.cond_norm = nn.GroupNorm(1, self.dim_cond)

        # Downsample
        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(zip(dims_h[:-1], dims_h[1:]))
        n_layers = len(in_out)

        down_layers = []
        for i in range(n_layers):
            layer_dim_in, layer_dim_out = in_out[i]
            resnet_depth = resnet_depths[i]
            attn_depth = attn_depths[i]
            down_layers.append(
                nn.ModuleList(
                    [
                        resnet_block_attn(
                            layer_dim_in,
                            layer_dim_out,
                            self.dim_emb,
                        ),
                        nn.ModuleList(
                            [
                                ResidualBlock(
                                    layer_dim_out,
                                    layer_dim_out,
                                    self.dim_emb,
                                    use_gca=attn_use_global_context_attention,
                                )
                                for _ in range(resnet_depth)
                            ],
                        ),
                        Transformer(
                            layer_dim_out,
                            self.dim_cond,
                            dim_head=attn_dim_head,
                            heads=attn_heads,
                            depth=attn_depth,
                            dropout=attn_dropout,
                            sdpa=attn_sdpa,
                            use_rotary_emb=attn_use_rotary_emb,
                        ),
                        Downsample(layer_dim_out, layer_dim_out)
                        if i < (n_layers - 1)
                        else nn.Conv1d(layer_dim_out, layer_dim_out, 3, padding=1),
                    ],
                ),
            )
        self.down_layers = nn.ModuleList(down_layers)

        # Middle
        self.middle_resnet1 = ResidualBlock(
            dims_h[-1],
            dims_h[-1],
            self.dim_emb,
            use_gca=attn_use_global_context_attention,
        )
        self.middle_transformer = Transformer(
            dims_h[-1],
            self.dim_cond,
            dim_head=attn_dim_head,
            heads=attn_heads,
            depth=attn_depth,
            dropout=attn_dropout,
            sdpa=attn_sdpa,
            use_rotary_emb=attn_use_rotary_emb,
        )
        self.middle_resnet2 = ResidualBlock(
            dims_h[-1],
            dims_h[-1],
            self.dim_emb,
            use_gca=attn_use_global_context_attention,
        )

        # Upsample
        resnet_depths = tuple(reversed(resnet_depths))
        attn_depths = tuple(reversed(attn_depths))
        in_out = tuple(reversed(tuple(zip(dims_h[:-1], dims_h[1:]))))
        n_layers = len(in_out)

        up_layers = []
        for i in range(n_layers):
            layer_dim_out, layer_dim_in = in_out[i]
            resnet_depth = resnet_depths[i]
            attn_depth = attn_depths[i]
            up_layers.append(
                nn.ModuleList(
                    [
                        resnet_block_attn(
                            layer_dim_in * 2,
                            layer_dim_out,
                            self.dim_emb,
                        ),
                        nn.ModuleList(
                            [
                                ResidualBlock(
                                    layer_dim_out,
                                    layer_dim_out,
                                    self.dim_emb,
                                    use_gca=attn_use_global_context_attention,
                                )
                                for _ in range(resnet_depth)
                            ],
                        ),
                        Transformer(
                            layer_dim_out,
                            self.dim_cond,
                            dim_head=attn_dim_head,
                            heads=attn_heads,
                            depth=attn_depth,
                            dropout=attn_dropout,
                            sdpa=attn_sdpa,
                            use_rotary_emb=attn_use_rotary_emb,
                        ),
                        Upsample(layer_dim_out, layer_dim_out)
                        if i < (n_layers - 1)
                        else nn.Conv1d(layer_dim_out, layer_dim_out, 3, padding=1),
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

        t = self.time_mlp(t)

        cond_mask = prob_mask_like((x.shape[0],), 1.0 - cond_drop_prob, device=x.device)
        cond_mask = rearrange(cond_mask, "b -> b 1")
        null_conds = repeat(self.null_cond, "d -> b d", b=x.shape[0])
        c = torch.where(cond_mask, c, null_conds)
        c = self.context_mlp(c)
        c = rearrange(c, "b d -> b d 1")
        c = self.cond_norm(c)

        skip_connections = []
        for init_down_block, down_blocks, down_transformer, downsample in self.down_layers:
            x = init_down_block(x, t, c)
            for down_resnet in down_blocks:
                x = down_resnet(x, t)
            x = down_transformer(x, c)
            skip_connections.append(x)
            x = downsample(x)

        x = self.middle_resnet1(x, t)
        x = self.middle_transformer(x, c)
        x = self.middle_resnet2(x, t)

        for init_up_block, up_blocks, up_transformer, upsample in self.up_layers:
            x = torch.cat([x, skip_connections.pop()], dim=1)
            x = init_up_block(x, t, c)
            for up_resnet in up_blocks:
                x = up_resnet(x, t)
            x = up_transformer(x, c)
            x = upsample(x)

        return self.final_conv(x)
