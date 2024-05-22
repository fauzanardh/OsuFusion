import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch.nn import functional as F  # noqa: N812

from osu_fusion.modules.attention import Attention
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


class FeedForward(nn.Sequential):
    def __init__(self: "FeedForward", dim: int, dim_mult: int = 4) -> None:
        inner_dim = dim * dim_mult
        super().__init__(
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, dim),
        )


class MultiHeadRMSNorm(nn.Module):
    def __init__(self: "MultiHeadRMSNorm", dim: int, heads: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self: "MultiHeadRMSNorm", x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class CMAttention(nn.Module):
    def __init__(
        self: "CMAttention",
        dim: int,
        heads: int,
        qk_norm: bool = True,
        causal: bool = True,
        use_rotary_emb: bool = True,
        infini: bool = True,
        segment_len: int = 1024,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.qk_norm = qk_norm

        self.to_qkv_x = nn.Linear(dim, self.dim_head * heads * 3, bias=False)
        self.q_x_norm = MultiHeadRMSNorm(self.dim_head, heads) if qk_norm else None
        self.k_x_norm = MultiHeadRMSNorm(self.dim_head, heads) if qk_norm else None

        self.to_qkv_a = nn.Linear(dim, self.dim_head * heads * 3, bias=False)
        self.q_a_norm = MultiHeadRMSNorm(self.dim_head, heads) if qk_norm else None
        self.k_a_norm = MultiHeadRMSNorm(self.dim_head, heads) if qk_norm else None

        self.attn = Attention(
            self.dim_head,
            heads=heads,
            causal=causal,
            use_rotary_emb=use_rotary_emb,
            infini=infini,
            segment_len=segment_len,
        )

    def forward(self: "CMAttention", x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
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

        out = self.attn(q, k, v)
        out_a, out_x = unpack(out, seq_shape, "b h * d")

        out_x, out_a = (rearrange(t, "b h n d -> b n (h d)") for t in (out_x, out_a))
        return out_x, out_a


class MMDiTBlock(nn.Module):
    def __init__(
        self: "MMDiTBlock",
        dim_h: int,
        dim_h_mult: int = 4,
        attn_heads: int = 8,
        attn_qk_norm: bool = True,
        attn_causal: bool = True,
        attn_use_rotary_emb: bool = True,
        attn_infini: bool = True,
        attn_segment_len: int = 1024,
    ) -> None:
        super().__init__()
        # Modulation
        self.modulation_x = nn.Sequential(
            nn.SiLU(),
            zero_init(nn.Linear(dim_h, dim_h * 6, bias=True)),
        )
        self.modulation_a = nn.Sequential(
            nn.SiLU(),
            zero_init(nn.Linear(dim_h, dim_h * 6, bias=True)),
        )

        # OsuData branch
        self.norm1_x = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.attn_out_x = nn.Linear(dim_h, dim_h, bias=False)
        self.norm2_x = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.mlp_x = FeedForward(dim_h, dim_mult=dim_h_mult)

        # Audio branch
        self.norm1_a = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.attn_out_a = nn.Linear(dim_h, dim_h, bias=False)
        self.norm2_a = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.mlp_a = FeedForward(dim_h, dim_mult=dim_h_mult)

        # Cross-Modal Attention (I don't know if the name is correct)
        self.attn = CMAttention(
            dim_h,
            attn_heads,
            qk_norm=attn_qk_norm,
            causal=attn_causal,
            use_rotary_emb=attn_use_rotary_emb,
            infini=attn_infini,
            segment_len=attn_segment_len,
        )

        self.gradient_checkpointing = False

    def forward_body(self: "MMDiTBlock", x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
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
        attn_out_x, attn_out_a = self.attn(h_x, h_a)

        x = x + gate_attn_x.unsqueeze(1) * (self.attn_out_x(attn_out_x))
        a = a + gate_attn_a.unsqueeze(1) * (self.attn_out_a(attn_out_a))

        # MLP
        x = x + gate_mlp_x.unsqueeze(1) * self.mlp_x(modulate(self.norm2_x(x), shift_mlp_x, scale_mlp_x))
        a = a + gate_mlp_a.unsqueeze(1) * self.mlp_a(modulate(self.norm2_a(a), shift_mlp_a, scale_mlp_a))

        return x, a

    def forward(self: "MMDiTBlock", x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, a, c, use_reentrant=True)
        else:
            return self.forward_body(x, a, c)


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


class FinalLayer(nn.Module):
    def __init__(self: "FinalLayer", dim_h: int, dim_out: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            zero_init(nn.Linear(dim_h, dim_h * 2, bias=True)),
        )
        self.linear = zero_init(nn.Linear(dim_h, dim_out))

    def forward(self: "FinalLayer", x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class MMDiT(nn.Module):
    def __init__(
        self: "MMDiT",
        dim_in_x: int,
        dim_in_a: int,
        dim_in_c: int,
        dim_h: int,
        dim_h_mult: int = 4,
        depth: int = 12,
        cross_embed_kernel_sizes: Tuple[int] = (3, 5, 7),
        attn_heads: int = 8,
        attn_qk_norm: bool = True,
        attn_causal: bool = True,
        attn_use_rotary_emb: bool = True,
        attn_infini: bool = True,
        attn_segment_len: int = 1024,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.attn_segment_len = attn_segment_len

        self.init_x = nn.Sequential(
            CrossEmbedLayer(dim_in_x, dim_h, cross_embed_kernel_sizes),
            Rearrange("b d n -> b n d"),
            nn.Linear(dim_h, dim_h),
        )
        self.init_a = nn.Sequential(
            CrossEmbedLayer(dim_in_a, dim_h, cross_embed_kernel_sizes),
            Rearrange("b d n -> b n d"),
            nn.Linear(dim_h, dim_h),
        )

        self.mlp_a = FeedForward(dim_h, dim_mult=dim_h_mult)
        self.feature_extractor_a = nn.Linear(dim_h * 2, dim_h)
        self.pos_emb_time = SinusoidalPositionEmbedding(dim_h)
        self.mlp_time = FeedForward(dim_h, dim_mult=dim_h_mult)
        self.mlp_cond = nn.Sequential(
            nn.Linear(dim_in_c, dim_h),
            FeedForward(dim_h, dim_mult=dim_h_mult),
        )
        self.null_cond = nn.Parameter(torch.randn(dim_h))

        self.blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim_h,
                    dim_h_mult=dim_h_mult,
                    attn_heads=attn_heads,
                    attn_qk_norm=attn_qk_norm,
                    attn_causal=attn_causal,
                    attn_use_rotary_emb=attn_use_rotary_emb,
                    attn_infini=attn_infini,
                    attn_segment_len=attn_segment_len,
                )
                for _ in range(depth)
            ],
        )

        self.final_layer = FinalLayer(dim_h, dim_in_x)

    def set_gradient_checkpointing(self: "MMDiT", value: bool) -> None:
        self.gradient_checkpointing = value
        for name, module in self.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
                print(f"Set gradient checkpointing to {value} for {name}")

    def forward_with_cond_scale(self: "MMDiT", *args: List, cond_scale: float = 1.0, **kwargs: Dict) -> torch.Tensor:
        logits = self(*args, **kwargs)

        if cond_scale == 1.0:
            return logits

        null_logits = self(*args, **kwargs, cond_drop_prob=1.0)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self: "MMDiT",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        # Pad to the closest multiple of attn_segment_len
        n = x.shape[-1]
        pad = (self.attn_segment_len - (n % self.attn_segment_len)) % self.attn_segment_len
        x = F.pad(x, (0, pad), value=-1.0)
        a = F.pad(a, (0, pad), value=0.0)

        x = self.init_x(x)
        a = self.init_a(a)

        cond_mask = prob_mask_like((x.shape[0],), 1.0 - cond_drop_prob, device=x.device)
        cond_mask = rearrange(cond_mask, "b -> b 1")
        null_conds = repeat(self.null_cond, "d -> b d", b=x.shape[0])
        c = self.mlp_cond(c)
        c = torch.where(cond_mask, c, null_conds)

        # Statistic audio features pooling
        mean_features_a = a.mean(dim=1)
        std_features_a = a.std(dim=1)
        h_a = torch.cat([mean_features_a, std_features_a], dim=1)
        h_a = self.feature_extractor_a(h_a)

        c = c + self.mlp_time(self.pos_emb_time(t)) + self.mlp_a(h_a)

        for block in self.blocks:
            x, a = block(x, a, c)

        x = self.final_layer(x, c)
        return rearrange(x, "b n d -> b d n")[:, :, :n]
