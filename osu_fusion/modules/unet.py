import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat

from osu_fusion.modules.residual import ResidualBlock
from osu_fusion.modules.transformer import TransformerBlock
from osu_fusion.modules.utils import prob_mask_like


def zero_init(module: nn.Module) -> nn.Module:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)

    return module


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self: "SinusoidalPositionEmbedding", dim: int, theta: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self: "SinusoidalPositionEmbedding", x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class CrossEmbedLayer(nn.Module):
    def __init__(self: "CrossEmbedLayer", dim: int, dim_out: int, kernel_sizes: Tuple[int]) -> None:
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        dim_scales = [int(dim / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        convs = []
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            convs.append(nn.Conv1d(dim, dim_scale, kernel, padding=kernel // 2))

        self.convs = nn.ModuleList(convs)

    def forward(self: "CrossEmbedLayer", x: torch.Tensor) -> torch.Tensor:
        return torch.cat([conv(x) for conv in self.convs], dim=1)


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


class Parallel(nn.Module):
    def __init__(self: "Parallel", *fns: nn.Module) -> None:
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self: "Parallel", x: torch.Tensor, *args: List, **kwargs: Dict) -> List[torch.Tensor]:
        return sum([fn(x, *args, **kwargs) for fn in self.fns])


class Residual(nn.Module):
    def __init__(self: "Residual", fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self: "Residual", x: torch.Tensor, *args: List, **kwargs: Dict) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self: "PreNorm", dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.fn = fn

    def forward(self: "PreNorm", x: torch.Tensor, *args: List, **kwargs: Dict) -> torch.Tensor:
        return self.fn(self.norm(x), *args, **kwargs)


class UNet(nn.Module):
    def __init__(
        self: "UNet",
        dim_in: int,
        dim_out: int,
        dim_h: int,
        dim_cond: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        num_layer_blocks: Tuple[int] = (2, 4, 8, 8),
        cross_embed_kernel_sizes: Tuple[int] = (3, 7, 15),
        attn_dim_head: int = 32,
        attn_heads: int = 8,
        attn_sdpa: bool = True,
        attn_use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.dim_emb = dim_h * 4
        self.dim_cond = dim_cond * 4

        self.init_conv = CrossEmbedLayer(dim_in, dim_h, cross_embed_kernel_sizes)
        self.final_resnet = ResidualBlock(dim_h * 2, dim_h, self.dim_emb, self.dim_cond)
        self.final_conv = zero_init(nn.Conv1d(dim_h, dim_out, 1))

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(self.dim_emb),
            nn.Linear(self.dim_emb, self.dim_emb),
            nn.SiLU(),
            nn.Linear(self.dim_emb, self.dim_emb),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(dim_cond, self.dim_cond),
            nn.SiLU(),
            nn.Linear(self.dim_cond, self.dim_cond),
        )
        self.null_cond = nn.Parameter(torch.randn(dim_cond))

        # Downsample
        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(zip(dims_h[:-1], dims_h[1:]))
        n_layers = len(in_out)

        down_layers = []
        for i in range(n_layers):
            layer_dim_in, layer_dim_out = in_out[i]
            num_blocks = num_layer_blocks[i]
            down_layers.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            layer_dim_in,
                            layer_dim_out,
                            self.dim_emb,
                            self.dim_cond,
                        ),
                        nn.ModuleList(
                            [
                                ResidualBlock(
                                    layer_dim_out,
                                    layer_dim_out,
                                    self.dim_emb,
                                    self.dim_cond,
                                )
                                for _ in range(num_blocks)
                            ],
                        ),
                        nn.ModuleList(
                            [
                                TransformerBlock(
                                    layer_dim_out,
                                    dim_head=attn_dim_head,
                                    heads=attn_heads,
                                    sdpa=attn_sdpa,
                                    use_rotary_emb=attn_use_rotary_emb,
                                )
                                for _ in range(num_blocks)
                            ],
                        ),
                        Downsample(layer_dim_out, layer_dim_out)
                        if i < (n_layers - 1)
                        else Parallel(
                            nn.Conv1d(layer_dim_out, layer_dim_out, 3, padding=1),
                            nn.Conv1d(layer_dim_out, layer_dim_out, 1),
                        ),
                    ],
                ),
            )
        self.down_layers = nn.ModuleList(down_layers)

        # Middle
        self.middle_resnet1 = ResidualBlock(
            dims_h[-1],
            dims_h[-1],
            self.dim_emb,
            self.dim_cond,
        )
        self.middle_transformers = nn.Sequential(
            *[
                TransformerBlock(
                    dims_h[-1],
                    dim_head=attn_dim_head,
                    heads=attn_heads,
                    sdpa=attn_sdpa,
                    use_rotary_emb=attn_use_rotary_emb,
                )
                for _ in range(num_blocks)
            ],
        )
        self.middle_resnet2 = ResidualBlock(
            dims_h[-1],
            dims_h[-1],
            self.dim_emb,
            self.dim_cond,
        )

        # Upsample
        in_out = tuple(reversed(tuple(zip(dims_h[:-1], dims_h[1:]))))
        num_layer_blocks = tuple(reversed(num_layer_blocks))
        n_layers = len(in_out)

        up_layers = []
        for i in range(n_layers):
            layer_dim_out, layer_dim_in = in_out[i]
            num_blocks = num_layer_blocks[i]
            up_layers.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            layer_dim_in * 2,
                            layer_dim_out,
                            self.dim_emb,
                            self.dim_cond,
                        ),
                        nn.ModuleList(
                            [
                                ResidualBlock(
                                    layer_dim_out,
                                    layer_dim_out,
                                    self.dim_emb,
                                    self.dim_cond,
                                )
                                for _ in range(num_blocks)
                            ],
                        ),
                        nn.ModuleList(
                            [
                                TransformerBlock(
                                    layer_dim_out,
                                    dim_head=attn_dim_head,
                                    heads=attn_heads,
                                    sdpa=attn_sdpa,
                                    use_rotary_emb=attn_use_rotary_emb,
                                )
                                for _ in range(num_blocks)
                            ],
                        ),
                        Upsample(layer_dim_out, layer_dim_out)
                        if i < (n_layers - 1)
                        else Parallel(
                            nn.Conv1d(layer_dim_out, layer_dim_out, 3, padding=1),
                            nn.Conv1d(layer_dim_out, layer_dim_out, 1),
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
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(t)

        cond_mask = prob_mask_like((x.shape[0],), 1.0 - cond_drop_prob, device=x.device)
        cond_mask = rearrange(cond_mask, "b -> b 1")
        null_conds = repeat(self.null_cond, "d -> b d", b=x.shape[0])
        c = torch.where(cond_mask, c, null_conds)
        c = self.cond_mlp(c)

        skip_connections = []
        for init_down_resnet, down_resnets, down_transformers, downsample in self.down_layers:
            x = init_down_resnet(x, t, c)
            for down_resnet, down_transformer in zip(down_resnets, down_transformers):
                x = down_resnet(x, t, c)
                x = down_transformer(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.middle_resnet1(x, t, c)
        x = self.middle_transformers(x)
        x = self.middle_resnet2(x, t, c)

        for init_up_resnet, up_resnets, up_transformers, upsample in self.up_layers:
            x = torch.cat([x, skip_connections.pop()], dim=1)
            x = init_up_resnet(x, t, c)
            for up_resnet, up_transformer in zip(up_resnets, up_transformers):
                x = up_resnet(x, t, c)
                x = up_transformer(x)
            x = upsample(x)

        x = torch.cat([x, r], dim=1)
        x = self.final_resnet(x, t, c)
        return self.final_conv(x)
