import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F  # noqa: N812

from osu_fusion.modules.attention import Attention
from osu_fusion.modules.residual import ResidualBlock
from osu_fusion.modules.utils import prob_mask_like


def zero_init(module: nn.Module) -> nn.Module:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)

    return module


@torch.jit.script
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class MultiHeadRMSNorm(nn.Module):
    def __init__(self: "MultiHeadRMSNorm", dim: int, heads: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self: "MultiHeadRMSNorm", x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class TransformerBlock(nn.Module):
    def __init__(
        self: "TransformerBlock",
        dim_in: int,
        dim_head: int,
        heads: int,
        qk_norm: bool = True,
        causal: bool = False,
        use_rotary_emb: bool = True,
        context_len: int = 4096,
        infini: bool = True,
        segment_len: int = 1024,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.qk_norm = qk_norm

        self.to_qkv = nn.Linear(dim_in, dim_head * heads * 3, bias=False)
        self.q_norm = MultiHeadRMSNorm(dim_head, heads) if qk_norm else None
        self.k_norm = MultiHeadRMSNorm(dim_head, heads) if qk_norm else None

        self.attn = Attention(
            dim_head,
            heads=heads,
            causal=causal,
            use_rotary_emb=use_rotary_emb,
            context_len=context_len,
            infini=infini,
            segment_len=segment_len,
        )
        self.linear = nn.Linear(dim_head * heads, dim_in)

        self.gradient_checkpointing = False

    def forward_body(self: "TransformerBlock", x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b d n -> b n d")

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = self.attn(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return rearrange(x + self.linear(out), "b n d -> b d n")

    def forward(self: "TransformerBlock", x: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, use_reentrant=True)
        else:
            return self.forward_body(x)


class AudioEncoder(nn.Module):
    def __init__(
        self: "AudioEncoder",
        dim_in: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        num_layer_blocks: Tuple[int] = (3, 3, 3, 3),
        cross_embed_kernel_sizes: Tuple[int] = (3, 7, 15),
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_qk_norm: bool = True,
        attn_causal: bool = False,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 8192,
        attn_infini: bool = True,
        attn_segment_len: int = 1024,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.dim_cond = dim_h * 4
        self.attn_context_len = attn_context_len

        self.init_conv = CrossEmbedLayer(dim_in, dim_h, cross_embed_kernel_sizes)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(self.dim_cond),
            nn.Linear(self.dim_cond, self.dim_cond),
            nn.SiLU(),
            nn.Linear(self.dim_cond, self.dim_cond),
        )

        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(zip(dims_h[:-1], dims_h[1:]))
        n_layers = len(in_out)

        layers = []
        for i in range(n_layers):
            layer_dim_in, layer_dim_out = in_out[i]
            num_blocks = num_layer_blocks[i]
            attn_context_len_layer = attn_context_len // (2**i)
            attn_segment_len_layer = attn_segment_len // (2**i)
            layers.append(
                nn.ModuleList(
                    [
                        ResidualBlock(layer_dim_in, layer_dim_out, self.dim_cond),
                        nn.ModuleList(
                            [ResidualBlock(layer_dim_out, layer_dim_out, self.dim_cond) for _ in range(num_blocks)],
                        ),
                        nn.ModuleList(
                            [
                                TransformerBlock(
                                    layer_dim_out,
                                    attn_dim_head,
                                    heads=attn_heads,
                                    qk_norm=attn_qk_norm,
                                    causal=attn_causal,
                                    use_rotary_emb=attn_use_rotary_emb,
                                    context_len=attn_context_len_layer,
                                    infini=attn_infini,
                                    segment_len=attn_segment_len_layer,
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
        self.layers = nn.ModuleList(layers)

    def forward(self: "AudioEncoder", x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)

        # Fixed time embedding (to reuse the same resnet module)
        b = x.shape[0]
        t = torch.zeros(b, dtype=torch.long, device=x.device)
        c = self.time_mlp(t)
        for init_resnet, resnets, transformers, downsample in self.layers:
            x = init_resnet(x, c)
            for resnet, transformer in zip(resnets, transformers):
                x = resnet(x, c)
                x = transformer(x)
            x = downsample(x)
        return x


class UNet(nn.Module):
    def __init__(
        self: "UNet",
        dim_in_x: int,
        dim_in_a: int,
        dim_in_c: int,
        dim_h: int,
        dim_h_mult: Tuple[int] = (1, 2, 3, 4),
        num_layer_blocks: Tuple[int] = (3, 3, 3, 3),
        num_middle_transformers: int = 3,
        cross_embed_kernel_sizes: Tuple[int] = (3, 7, 15),
        attn_dim_head: int = 32,
        attn_heads: int = 8,
        attn_qk_norm: bool = True,
        attn_causal: bool = False,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 8192,
        attn_infini: bool = True,
        attn_segment_len: int = 1024,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.dim_cond = dim_h * 4
        self.attn_context_len = attn_context_len
        self.attn_infini = attn_infini
        self.attn_segment_len = attn_segment_len

        self.init_x = CrossEmbedLayer(dim_in_x, dim_h, cross_embed_kernel_sizes)
        self.audio_encoder = AudioEncoder(
            dim_in_a,
            dim_h,
            dim_h_mult=dim_h_mult,
            num_layer_blocks=num_layer_blocks,
            cross_embed_kernel_sizes=cross_embed_kernel_sizes,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_qk_norm=attn_qk_norm,
            attn_causal=attn_causal,
            attn_use_rotary_emb=attn_use_rotary_emb,
            attn_infini=attn_infini,
            attn_segment_len=attn_segment_len,
        )
        self.final_resnet = ResidualBlock(dim_h * 2, dim_h, self.dim_cond)
        self.final_conv = zero_init(nn.Conv1d(dim_h, dim_in_x, 1))

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(self.dim_cond),
            nn.Linear(self.dim_cond, self.dim_cond),
            nn.SiLU(),
            nn.Linear(self.dim_cond, self.dim_cond),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(dim_in_c, self.dim_cond),
            nn.SiLU(),
            nn.Linear(self.dim_cond, self.dim_cond),
        )
        self.null_cond = nn.Parameter(torch.randn(dim_in_c))

        # Downsample
        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(zip(dims_h[:-1], dims_h[1:]))
        n_layers = len(in_out)

        down_layers = []
        for i in range(n_layers):
            layer_dim_in, layer_dim_out = in_out[i]
            num_blocks = num_layer_blocks[i]
            attn_context_len_layer = attn_context_len // (2**i)
            attn_segment_len_layer = attn_segment_len // (2**i)
            down_layers.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            layer_dim_in,
                            layer_dim_out,
                            self.dim_cond,
                        ),
                        nn.ModuleList(
                            [
                                ResidualBlock(
                                    layer_dim_out,
                                    layer_dim_out,
                                    self.dim_cond,
                                )
                                for _ in range(num_blocks)
                            ],
                        ),
                        nn.ModuleList(
                            [
                                TransformerBlock(
                                    layer_dim_out,
                                    attn_dim_head,
                                    heads=attn_heads,
                                    qk_norm=attn_qk_norm,
                                    causal=attn_causal,
                                    use_rotary_emb=attn_use_rotary_emb,
                                    context_len=attn_context_len_layer,
                                    infini=attn_infini,
                                    segment_len=attn_segment_len_layer,
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
            dims_h[-1] * 2,
            dims_h[-1],
            self.dim_cond,
        )
        self.middle_transformer = nn.ModuleList(
            [
                TransformerBlock(
                    dims_h[-1],
                    attn_dim_head,
                    heads=attn_heads,
                    qk_norm=attn_qk_norm,
                    causal=attn_causal,
                    use_rotary_emb=attn_use_rotary_emb,
                    context_len=attn_context_len // (2 ** (n_layers - 1)),
                    infini=attn_infini,
                    segment_len=attn_segment_len,
                )
                for _ in range(num_middle_transformers)
            ],
        )
        self.middle_resnet2 = ResidualBlock(
            dims_h[-1],
            dims_h[-1],
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
            attn_context_len_layer = attn_context_len // (2 ** (n_layers - i - 1))
            attn_segment_len_layer = attn_segment_len // (2 ** (n_layers - i - 1))
            up_layers.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            layer_dim_in * 2,
                            layer_dim_out,
                            self.dim_cond,
                        ),
                        nn.ModuleList(
                            [
                                ResidualBlock(
                                    layer_dim_out,
                                    layer_dim_out,
                                    self.dim_cond,
                                )
                                for _ in range(num_blocks)
                            ],
                        ),
                        nn.ModuleList(
                            [
                                TransformerBlock(
                                    layer_dim_out,
                                    attn_dim_head,
                                    heads=attn_heads,
                                    qk_norm=attn_qk_norm,
                                    causal=attn_causal,
                                    use_rotary_emb=attn_use_rotary_emb,
                                    context_len=attn_context_len_layer,
                                    infini=attn_infini,
                                    segment_len=attn_segment_len_layer,
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

    def set_gradient_checkpointing(self: "UNet", value: bool) -> None:
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
        n = x.shape[-1]
        if self.attn_infini:
            # Pad to the multiple of attn_segment_len
            segment_len = self.attn_segment_len
            pad_len = (segment_len - (n % segment_len)) % segment_len
        else:
            # Pad to the multiple of 2^unet depth
            depth = len(self.down_layers)
            pad_len = (2**depth - (n % (2**depth))) % (2**depth)
        x = F.pad(x, (0, pad_len), value=-1.0)
        a = F.pad(a, (0, pad_len), value=0.0)

        x = self.init_x(x)
        a = self.audio_encoder(a)

        r = x.clone()

        cond_mask = prob_mask_like((x.shape[0],), 1.0 - cond_drop_prob, device=x.device)
        cond_mask = rearrange(cond_mask, "b -> b 1")
        null_conds = repeat(self.null_cond, "d -> b d", b=x.shape[0])
        c = torch.where(cond_mask, c, null_conds)
        c = self.cond_mlp(c) + self.time_mlp(t)

        skip_connections = []
        for init_down_resnet, down_resnets, down_transformers, downsample in self.down_layers:
            x = init_down_resnet(x, c)
            for down_resnet, down_transformer in zip(down_resnets, down_transformers):
                x = down_resnet(x, c)
                x = down_transformer(x)
            skip_connections.append(x)
            x = downsample(x)

        x = torch.cat([x, a], dim=1)
        x = self.middle_resnet1(x, c)
        for transformer_block in self.middle_transformer:
            x = transformer_block(x)
        x = self.middle_resnet2(x, c)

        for init_up_resnet, up_resnets, up_transformers, upsample in self.up_layers:
            x = torch.cat([x, skip_connections.pop()], dim=1)
            x = init_up_resnet(x, c)
            for up_resnet, up_transformer in zip(up_resnets, up_transformers):
                x = up_resnet(x, c)
                x = up_transformer(x)
            x = upsample(x)

        x = torch.cat([x, r], dim=1)
        x = self.final_resnet(x, c)
        return self.final_conv(x)[:, :, :n]
