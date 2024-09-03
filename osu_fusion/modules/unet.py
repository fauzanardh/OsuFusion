import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F  # noqa: N812

from osu_fusion.modules.attention import Attend, RotaryPositionEmbedding
from osu_fusion.modules.residual import ResidualBlock
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


class Upsample(nn.Module):
    def __init__(self: "Upsample", dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, 3, padding=1)

    def forward(self: "Upsample", x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self: "Downsample", dim_in: int, dim_out: int) -> None:
        super().__init__()
        # Asymmetric padding
        self.conv = nn.Conv1d(dim_in, dim_out, 3, stride=2, padding=0)

    def forward(self: "Downsample", x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1)
        x = F.pad(x, pad=pad, mode="reflect")
        x = self.conv(x)
        return x


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


class Attention(nn.Module):
    def __init__(
        self: "Attention",
        dim_in: int,
        dim_head: int,
        heads: int,
        kv_heads: int,
        qk_norm: bool = True,
        use_rotary_emb: bool = True,
        context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.qk_norm = qk_norm
        self.use_rotary_emb = use_rotary_emb

        self.to_q = nn.Linear(dim_in, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim_in, dim_head * kv_heads * 2, bias=False)
        self.q_norm = MultiHeadRMSNorm(dim_head, heads) if qk_norm else None
        self.k_norm = MultiHeadRMSNorm(dim_head, kv_heads) if qk_norm else None
        if use_rotary_emb:
            self.rotary_emb = RotaryPositionEmbedding(dim_head, scale_base=context_len)

        self.attn = Attend()
        self.to_out = nn.Linear(dim_head * heads, dim_in)

    def forward(self: "Attention", x: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)

        k, v = self.to_kv(x).chunk(2, dim=-1)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.kv_heads) for t in (k, v))
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # GQA
        k, v = (repeat(t, "b h n d -> b (r h) n d", r=self.heads // self.kv_heads) for t in (k, v))

        if self.use_rotary_emb:
            q, k = self.rotary_emb(q, k)

        out = self.attn(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return x + self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(
        self: "LinearAttention",
        dim_in: int,
        dim_head: int,
        heads: int,
        kv_heads: int,
        qk_norm: bool = True,
        use_rotary_emb: bool = True,
        context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.qk_norm = qk_norm
        self.use_rotary_emb = use_rotary_emb

        self.to_q = nn.Linear(dim_in, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim_in, dim_head * kv_heads * 2, bias=False)
        self.q_norm = MultiHeadRMSNorm(dim_head, heads) if qk_norm else None
        self.k_norm = MultiHeadRMSNorm(dim_head, kv_heads) if qk_norm else None

        if self.use_rotary_emb:
            self.rotary_emb = RotaryPositionEmbedding(dim_head, scale_base=context_len)

        self.to_out = nn.Linear(dim_head * heads, dim_in)

    def forward(self: "LinearAttention", x: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)

        k, v = self.to_kv(x).chunk(2, dim=-1)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.kv_heads) for t in (k, v))
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # GQA
        if self.kv_heads > 1:
            k, v = (repeat(t, "b h n d -> b (r h) n d", r=self.heads // self.kv_heads) for t in (k, v))

        if self.use_rotary_emb:
            q, k = self.rotary_emb(q, k)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        context = torch.einsum("b h n d, b h n e -> b h d e", k, v)

        out = torch.einsum("b h d e, b h n d -> b h n e", context, q)
        out = rearrange(out, "b h n d -> b n (h d)")
        return x + self.to_out(out)


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
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 4096,
        attn_linear: bool = False,
    ) -> None:
        super().__init__()
        self.attn = (
            Attention(
                dim,
                attn_dim_head,
                attn_heads,
                attn_kv_heads,
                attn_qk_norm,
                attn_use_rotary_emb,
                attn_context_len,
            )
            if not attn_linear
            else LinearAttention(
                dim,
                attn_dim_head,
                attn_heads,
                attn_kv_heads,
                attn_qk_norm,
                attn_use_rotary_emb,
                attn_context_len,
            )
        )
        self.ff = FeedForward(dim, ff_mult)

    def forward(self: "TransformerBlock", x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b d n -> b n d")
        x = self.attn(x)
        x = self.ff(x) + x
        return rearrange(x, "b n d -> b d n")


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
        attn_qk_norm: bool,
        attn_use_rotary_emb: bool,
        attn_context_len: int,
    ) -> None:
        super().__init__()
        self.init_resnet = ResidualBlock(dim_in if down_block else dim_in + dim_out, dim_in, dim_time, dim_cond)
        self.resnets = nn.ModuleList(
            [
                ResidualBlock(dim_in if down_block else dim_in + dim_out, dim_in, dim_time, dim_cond)
                for _ in range(num_blocks)
            ],
        )
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(
                    dim_in,
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_kv_heads=attn_kv_heads,
                    attn_qk_norm=attn_qk_norm,
                    attn_use_rotary_emb=attn_use_rotary_emb,
                    attn_context_len=attn_context_len,
                    attn_linear=True,
                )
                for _ in range(num_blocks)
            ],
        )
        if down_block:
            self.sampler = (
                Downsample(dim_in, dim_out)
                if layer_idx < (num_layers - 1)
                else Parallel(
                    nn.Conv1d(dim_in, dim_out, 3, padding=1),
                    nn.Conv1d(dim_in, dim_out, 1),
                )
            )
        else:
            self.sampler = (
                Upsample(dim_in, dim_out)
                if layer_idx < (num_layers - 1)
                else Parallel(
                    nn.Conv1d(dim_in, dim_out, 3, padding=1),
                    nn.Conv1d(dim_in, dim_out, 1),
                )
            )

        self.gradient_checkpointing = False

    def forward_body(
        self: "UNetBlock",
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        skip_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        skips = []

        if skip_inputs is not None:
            assert (
                len(skip_inputs) == len(self.resnets) + 1
            ), f"Expected {len(self.resnets) + 1} skip inputs, got {len(skip_inputs)}"
            x = torch.cat([x, skip_inputs[0]], dim=1)
        x = self.init_resnet(x, t, c)
        skips.append(x)

        for i, (resnet, transformer) in enumerate(zip(self.resnets, self.transformers)):
            if skip_inputs is not None:
                x = torch.cat([x, skip_inputs[i + 1]], dim=1)
            x = resnet(x, t, c)
            x = transformer(x)
            skips.append(x)

        return self.sampler(x), skips

    def forward(
        self: "UNetBlock",
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        skip_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, t, c, skip_inputs, use_reentrant=True)
        else:
            return self.forward_body(x, t, c, skip_inputs=skip_inputs)


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
        attn_kv_heads: int = 2,
        attn_qk_norm: bool = True,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.dim_emb = dim_h * 4
        self.attn_context_len = attn_context_len

        self.init_conv = CrossEmbedLayer(dim_in, dim_h, cross_embed_kernel_sizes)

        dims_h = tuple((dim_h * mult) for mult in dim_h_mult)
        dims_h = (dim_h, *dims_h)
        in_out = tuple(zip(dims_h[:-1], dims_h[1:]))
        n_layers = len(in_out)

        layers = []
        for i in range(n_layers):
            layer_dim_in, layer_dim_out = in_out[i]
            num_blocks = num_layer_blocks[i]
            attn_context_len_layer = attn_context_len // (2**i)
            layers.append(
                UNetBlock(
                    layer_dim_in,
                    layer_dim_out,
                    None,
                    None,
                    i,
                    n_layers,
                    num_blocks,
                    True,
                    attn_dim_head,
                    attn_heads,
                    attn_kv_heads,
                    attn_qk_norm,
                    attn_use_rotary_emb,
                    attn_context_len_layer,
                ),
            )
        self.layers = nn.ModuleList(layers)

    def forward(self: "AudioEncoder", x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        for layer in self.layers:
            x, _ = layer(x)
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
        attn_dim_head: int = 64,
        attn_heads: int = 16,
        attn_kv_heads: int = 4,
        attn_qk_norm: bool = True,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.dim_emb = dim_h * 4
        self.attn_context_len = attn_context_len

        self.init_x = CrossEmbedLayer(dim_in_x, dim_h, cross_embed_kernel_sizes)
        self.audio_encoder = AudioEncoder(
            dim_in_a,
            dim_h,
            dim_h_mult=dim_h_mult,
            num_layer_blocks=num_layer_blocks,
            cross_embed_kernel_sizes=cross_embed_kernel_sizes,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            attn_kv_heads=attn_kv_heads,
            attn_qk_norm=attn_qk_norm,
            attn_use_rotary_emb=attn_use_rotary_emb,
        )
        self.final_resnet = ResidualBlock(dim_h * 2, dim_h, self.dim_emb, self.dim_emb)
        self.final_conv = zero_init(nn.Conv1d(dim_h, dim_in_x, 1))

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(self.dim_emb),
            nn.Linear(self.dim_emb, self.dim_emb),
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
        in_out = tuple(zip(dims_h[:-1], dims_h[1:]))
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
                    attn_qk_norm,
                    attn_use_rotary_emb,
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
                    attn_qk_norm=attn_qk_norm,
                    attn_use_rotary_emb=attn_use_rotary_emb,
                    attn_context_len=attn_context_len // (2 ** (n_layers - 1)),
                    attn_linear=False,
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
        in_out = tuple(reversed(tuple(zip(dims_h[:-1], dims_h[1:]))))
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
                    attn_qk_norm,
                    attn_use_rotary_emb,
                    attn_context_len_layer,
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
        # Pad to the multiple of 2^unet depth
        depth = len(self.down_layers)
        pad_len = (2**depth - (n % (2**depth))) % (2**depth)
        x = F.pad(x, (0, pad_len), value=-1.0)
        a = F.pad(a, (0, pad_len), value=-23.0)

        x = self.init_x(x)
        a = self.audio_encoder(a)
        t = self.time_mlp(t)

        r = x.clone()

        # Prepare condition
        cond_mask = prob_mask_like((x.shape[0],), 1.0 - cond_drop_prob, device=x.device)
        cond_mask = rearrange(cond_mask, "b -> b 1")
        null_conds = repeat(self.null_cond, "d -> b d", b=x.shape[0])
        c = self.cond_mlp(c)
        c = torch.where(cond_mask, c, null_conds)

        skips_connections = []
        for down_layer in self.down_layers:
            x, skips = down_layer(x, t, c)
            skips_connections.append(skips)

        x = torch.cat([x, a], dim=1)
        x = self.middle_resnet1(x, t, c)
        for transformer_block in self.middle_transformer:
            x = transformer_block(x)
        x = self.middle_resnet2(x, t, c)

        for up_layer, skips in zip(self.up_layers, reversed(skips_connections)):
            x, _ = up_layer(x, t, c, skip_inputs=skips)

        x = torch.cat([x, r], dim=1)
        x = self.final_resnet(x, t, c)

        return self.final_conv(x)[:, :, :n]
