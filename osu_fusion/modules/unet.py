import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import pack, rearrange, repeat, unpack
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


class LastLayer(nn.Module):
    def __init__(self: "LastLayer", dim: int, dim_out: int) -> None:
        super().__init__()
        self.out_x = Parallel(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.Conv1d(dim, dim_out, 1),
        )
        self.out_a = Parallel(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.Conv1d(dim, dim_out, 1),
        )

    def forward(self: "LastLayer", x: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.out_x(x), self.out_a(a)


class FeedForward(nn.Sequential):
    def __init__(self: "FeedForward", dim: int, dim_mult: int = 4) -> None:
        inner_dim = dim * dim_mult
        super().__init__(
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, dim),
        )
        self.gradient_checkpointing = False

    def forward(self: "FeedForward", x: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(super().forward, x, use_reentrant=False)
        else:
            return super().forward(x)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self: "MultiHeadRMSNorm", dim: int, heads: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self: "MultiHeadRMSNorm", x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class JointAttention(nn.Module):
    def __init__(
        self: "JointAttention",
        dim_in: int,
        dim_head: int,
        heads: int,
        qk_norm: bool = True,
        causal: bool = True,
        use_rotary_emb: bool = True,
        context_len: int = 4096,
        infini: bool = True,
        segment_len: int = 1024,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.qk_norm = qk_norm

        self.to_qkv_x = nn.Linear(dim_in, dim_head * heads * 3, bias=False)
        self.q_x_norm = MultiHeadRMSNorm(dim_head, heads) if qk_norm else None
        self.k_x_norm = MultiHeadRMSNorm(dim_head, heads) if qk_norm else None

        self.to_qkv_a = nn.Linear(dim_in, dim_head * heads * 3, bias=False)
        self.q_a_norm = MultiHeadRMSNorm(dim_head, heads) if qk_norm else None
        self.k_a_norm = MultiHeadRMSNorm(dim_head, heads) if qk_norm else None

        self.attn = Attention(
            dim_head,
            heads=heads,
            causal=causal,
            use_rotary_emb=use_rotary_emb,
            context_len=context_len,
            infini=infini,
            segment_len=segment_len,
        )

    def forward(
        self: "JointAttention",
        x: torch.Tensor,
        a: torch.Tensor,
        padding_data: Optional[Tuple[int, List[int]]] = None,
    ) -> torch.Tensor:
        q_x, k_x, v_x = self.to_qkv_x(x).chunk(3, dim=-1)
        q_x, k_x, v_x = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q_x, k_x, v_x))

        q_a, k_a, v_a = self.to_qkv_a(a).chunk(3, dim=-1)
        q_a, k_a, v_a = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q_a, k_a, v_a))

        if self.qk_norm:
            q_x = self.q_x_norm(q_x)
            k_x = self.k_x_norm(k_x)
            q_a = self.q_a_norm(q_a)
            k_a = self.k_a_norm(k_a)

        # Combine the audio data first and then the osu data
        # logic behind it is that we let the model learn the audio data first
        # and then the osu data can attend to the audio data
        q, seq_shape = pack([q_a, q_x], "b h * d")
        k, _ = pack([k_a, k_x], "b h * d")
        v, _ = pack([v_a, v_x], "b h * d")

        out = self.attn(q, k, v, padding_data=padding_data)
        out_a, out_x = unpack(out, seq_shape, "b h * d")

        out_x, out_a = (rearrange(t, "b h n d -> b n (h d)") for t in (out_x, out_a))
        return out_x, out_a


class MMDiTBlock(nn.Module):
    def __init__(
        self: "MMDiTBlock",
        dim_h: int,
        dim_cond: int,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_qk_norm: bool = True,
        attn_causal: bool = True,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 4096,
        attn_infini: bool = True,
        attn_segment_len: int = 1024,
    ) -> None:
        super().__init__()
        # Modulation
        self.modulation_x = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_cond, dim_h * 6, bias=True),
        )
        self.modulation_a = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_cond, dim_h * 6, bias=True),
        )

        # OsuData branch
        self.norm1_x = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.attn_out_x = nn.Linear(attn_dim_head * attn_heads, dim_h, bias=False)
        self.norm2_x = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.mlp_x = FeedForward(dim_h)

        # Audio branch
        self.norm1_a = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.attn_out_a = nn.Linear(attn_dim_head * attn_heads, dim_h, bias=False)
        self.norm2_a = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.mlp_a = FeedForward(dim_h)

        self.attn = JointAttention(
            dim_h,
            attn_dim_head,
            attn_heads,
            qk_norm=attn_qk_norm,
            causal=attn_causal,
            use_rotary_emb=attn_use_rotary_emb,
            context_len=attn_context_len,
            infini=attn_infini,
            segment_len=attn_segment_len,
        )

        self.gradient_checkpointing = False

    def forward_body(
        self: "MMDiTBlock",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
        padding_data: Optional[Tuple[int, List[int]]] = None,
    ) -> torch.Tensor:
        # Rearrange data for attention
        x = rearrange(x, "b d n -> b n d")
        a = rearrange(a, "b d n -> b n d")

        # Modulation
        (
            shift_attn_x,
            scale_attn_x,
            gate_attn_x,
            shift_mlp_x,
            scale_mlp_x,
            gate_mlp_x,
        ) = self.modulation_x(c).chunk(6, dim=1)
        (
            shift_attn_a,
            scale_attn_a,
            gate_attn_a,
            shift_mlp_a,
            scale_mlp_a,
            gate_mlp_a,
        ) = self.modulation_a(c).chunk(6, dim=1)

        # Attention
        h_x = modulate(self.norm1_x(x), shift_attn_x, scale_attn_x)
        h_a = modulate(self.norm1_a(a), shift_attn_a, scale_attn_a)
        attn_out_x, attn_out_a = self.attn(h_x, h_a, padding_data=padding_data)

        x = x + gate_attn_x.unsqueeze(1) * (self.attn_out_x(attn_out_x))
        a = a + gate_attn_a.unsqueeze(1) * (self.attn_out_a(attn_out_a))

        # MLP
        x = x + gate_mlp_x.unsqueeze(1) * self.mlp_x(modulate(self.norm2_x(x), shift_mlp_x, scale_mlp_x))
        a = a + gate_mlp_a.unsqueeze(1) * self.mlp_a(modulate(self.norm2_a(a), shift_mlp_a, scale_mlp_a))

        return rearrange(x, "b n d -> b d n"), rearrange(a, "b n d -> b d n")

    def forward(
        self: "MMDiTBlock",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
        padding_data: Optional[Tuple[int, List[int]]] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, a, c, padding_data, use_reentrant=True)
        else:
            return self.forward_body(x, a, c, padding_data=padding_data)


class TransformerBlock(nn.Module):
    def __init__(
        self: "TransformerBlock",
        dim_in: int,
        dim_head: int,
        heads: int,
        qk_norm: bool = True,
        causal: bool = True,
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
        attn_causal: bool = True,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 4096,
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
        self.middle_resnet1 = ResidualBlock(dims_h[-1], dims_h[-1], self.dim_cond)
        self.middle_transformers = nn.Sequential(
            *[
                TransformerBlock(
                    layer_dim_out,
                    attn_dim_head,
                    heads=attn_heads,
                    qk_norm=attn_qk_norm,
                    causal=attn_causal,
                    use_rotary_emb=attn_use_rotary_emb,
                    context_len=attn_context_len // (2 ** (n_layers - 1)),
                    infini=attn_infini,
                    segment_len=attn_segment_len // (2 ** (n_layers - 1)),
                )
                for _ in range(num_blocks)
            ],
        )
        self.middle_resnet2 = ResidualBlock(dims_h[-1], dims_h[-1], self.dim_cond)

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

        x = self.middle_resnet1(x, c)
        x = self.middle_transformers(x)
        x = self.middle_resnet2(x, c)
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
        num_mmdit_blocks: int = 6,
        cross_embed_kernel_sizes: Tuple[int] = (3, 7, 15),
        attn_dim_head: int = 32,
        attn_heads: int = 8,
        attn_qk_norm: bool = True,
        attn_causal: bool = True,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 4096,
        attn_infini: bool = True,
        attn_segment_len: int = 1024,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.dim_cond = dim_h * 4
        self.attn_context_len = attn_context_len
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
            dims_h[-1],
            dims_h[-1],
            self.dim_cond,
        )
        self.middle_mmdit = nn.ModuleList(
            [
                MMDiTBlock(
                    dims_h[-1],
                    self.dim_cond,
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_qk_norm=attn_qk_norm,
                    attn_causal=attn_causal,
                    attn_use_rotary_emb=attn_use_rotary_emb,
                    attn_context_len=(attn_context_len // (2 ** (n_layers - 1))) * 2,  # 2x since we concatenate x and a
                    attn_infini=attn_infini,
                    attn_segment_len=attn_segment_len // (2 ** (n_layers - 1)),
                )
                for _ in range(num_mmdit_blocks)
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
        segment_len = self.attn_segment_len // 2
        pad_len = (segment_len - (n % segment_len)) % segment_len
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

        x = self.middle_resnet1(x, c)
        for mmdit in self.middle_mmdit:
            x, a = mmdit(x, a, c)
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
