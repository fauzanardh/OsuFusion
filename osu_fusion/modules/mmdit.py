import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F  # noqa: N812

from osu_fusion.modules.attention import Attention, RotaryPositionEmbedding
from osu_fusion.modules.utils import prob_mask_like


def zero_init(module: nn.Module) -> nn.Module:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)

    return module


@torch.jit.script
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


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


class CMAttention(nn.Module):
    def __init__(
        self: "CMAttention",
        dim: int,
        heads: int,
        sdpa: bool = True,
        qk_norm: bool = True,
        one_kv: bool = True,
        use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.one_kv = one_kv
        self.qk_norm = qk_norm

        self.q_x_norm_scale = nn.Parameter(torch.ones(heads, 1, self.dim_head)) if qk_norm else None
        self.q_a_norm_scale = nn.Parameter(torch.ones(heads, 1, self.dim_head)) if qk_norm else None
        if one_kv:
            self.to_q_x = nn.Linear(dim, dim, bias=False)
            self.to_kv_x = nn.Linear(dim, self.dim_head * 2, bias=False)
            self.to_q_a = nn.Linear(dim, dim, bias=False)
            self.to_kv_a = nn.Linear(dim, self.dim_head * 2, bias=False)

            self.k_x_norm_scale = nn.Parameter(torch.ones(1, 1, self.dim_head)) if qk_norm else None
            self.k_a_norm_scale = nn.Parameter(torch.ones(1, 1, self.dim_head)) if qk_norm else None
        else:
            self.to_qkv_x = nn.Linear(dim, dim * 3, bias=False)
            self.to_qkv_a = nn.Linear(dim, dim * 3, bias=False)

            self.k_x_norm_scale = nn.Parameter(torch.ones(heads, 1, self.dim_head)) if qk_norm else None
            self.k_a_norm_scale = nn.Parameter(torch.ones(heads, 1, self.dim_head)) if qk_norm else None

        self.rotary_emb = RotaryPositionEmbedding(self.dim_head * 2) if use_rotary_emb else None
        self.attn = Attention(sdpa=sdpa)

    def forward(self: "CMAttention", x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if self.one_kv:
            q_x = self.to_q_x(x)
            q_x = rearrange(q_x, "b n (h d) -> b h n d", h=self.heads)
            q_a = self.to_q_a(a)
            q_a = rearrange(q_a, "b n (h d) -> b h n d", h=self.heads)

            k_x, v_x = self.to_kv_x(x).chunk(2, dim=-1)
            k_x, v_x = (rearrange(t, "b n d -> b 1 n d") for t in (k_x, v_x))
            k_a, v_a = self.to_kv_a(a).chunk(2, dim=-1)
            k_a, v_a = (rearrange(t, "b n d -> b 1 n d") for t in (k_a, v_a))
        else:
            q_x, k_x, v_x = self.to_qkv_x(x).chunk(3, dim=-1)
            q_x, k_x, v_x = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q_x, k_x, v_x))
            q_a, k_a, v_a = self.to_qkv_a(a).chunk(3, dim=-1)
            q_a, k_a, v_a = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q_a, k_a, v_a))

        if self.qk_norm:
            q_x, k_x, q_a, k_a = map(l2norm, (q_x, k_x, q_a, k_a))
            q_x = q_x * self.q_x_norm_scale
            k_x = k_x * self.k_x_norm_scale
            q_a = q_a * self.q_a_norm_scale
            k_a = k_a * self.k_a_norm_scale

        q = torch.cat([q_x, q_a], dim=-1)
        k = torch.cat([k_x, k_a], dim=-1)
        v = torch.cat([v_x, v_a], dim=-1)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)

        out = self.attn(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return out


class MMDiTBlock(nn.Module):
    def __init__(
        self: "MMDiTBlock",
        dim_h: int,
        dim_h_mult: int = 4,
        attn_heads: int = 6,
        attn_sdpa: bool = True,
        attn_qk_norm: bool = True,
        attn_one_kv: bool = True,
        attn_use_rotary_emb: bool = True,
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
        self.attn_out_x = nn.Linear(dim_h * 2, dim_h, bias=False)
        self.norm2_x = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.mlp_x = FeedForward(dim_h, dim_mult=dim_h_mult)

        # Audio branch
        self.norm1_a = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.attn_out_a = nn.Linear(dim_h * 2, dim_h, bias=False)
        self.norm2_a = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.mlp_a = FeedForward(dim_h, dim_mult=dim_h_mult)

        # Cross-Modal Attention (I don't know if the name is correct)
        self.attn = CMAttention(
            dim_h,
            attn_heads,
            sdpa=attn_sdpa,
            qk_norm=attn_qk_norm,
            one_kv=attn_one_kv,
            use_rotary_emb=attn_use_rotary_emb,
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
        attn_out = self.attn(h_x, h_a)

        x = x + gate_attn_x.unsqueeze(1) * self.attn_out_x(attn_out)
        a = a + gate_attn_a.unsqueeze(1) * self.attn_out_a(attn_out)

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
        attn_heads: int = 6,
        attn_sdpa: bool = True,
        attn_qk_norm: bool = True,
        attn_one_kv: bool = True,
        attn_use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.dim_h = dim_h

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
        self.mlp_time = nn.Sequential(
            SinusoidalPositionEmbedding(dim_h),
            FeedForward(dim_h, dim_mult=dim_h_mult),
        )
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
                    attn_sdpa=attn_sdpa,
                    attn_qk_norm=attn_qk_norm,
                    attn_one_kv=attn_one_kv,
                    attn_use_rotary_emb=attn_use_rotary_emb,
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

        c = c + self.mlp_time(t) + self.mlp_a(h_a)

        for block in self.blocks:
            x, a = block(x, a, c)

        x = self.final_layer(x, c)
        return rearrange(x, "b n d -> b d n")
