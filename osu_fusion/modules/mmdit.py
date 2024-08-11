import math
from typing import Dict, List

import torch
import torch.nn as nn
from einops import pack, rearrange, repeat, unpack
from torch.nn import functional as F  # noqa: N812

from osu_fusion.modules.attention import Attend
from osu_fusion.modules.utils import prob_mask_like


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


class PatchEmbedding(nn.Module):
    def __init__(self: "PatchEmbedding", dim_in: int, dim_emb: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(dim_in, dim_emb, patch_size, stride=patch_size)

    def forward(self: "PatchEmbedding", x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % self.patch_size == 0, "Input sequence length must be divisible by the patch size"
        return rearrange(self.proj(x), "b d n -> b n d")


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
        dim: int,
        dim_head: int,
        heads: int,
        kv_heads: int,
        qk_norm: bool = True,
        causal: bool = True,
        use_rotary_emb: bool = True,
        context_len: int = 8192,
        infini: bool = True,
        segment_len: int = 1024,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.qk_norm = qk_norm

        self.to_q_x = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k_x = nn.Linear(dim, dim_head * kv_heads, bias=False)
        self.to_v_x = nn.Linear(dim, dim_head * kv_heads, bias=False)
        self.q_x_norm = MultiHeadRMSNorm(dim_head, heads) if qk_norm else nn.Identity()
        self.k_x_norm = MultiHeadRMSNorm(dim_head, kv_heads) if qk_norm else nn.Identity()

        self.to_q_a = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k_a = nn.Linear(dim, dim_head * kv_heads, bias=False)
        self.to_v_a = nn.Linear(dim, dim_head * kv_heads, bias=False)
        self.q_a_norm = MultiHeadRMSNorm(dim_head, heads) if qk_norm else nn.Identity()
        self.k_a_norm = MultiHeadRMSNorm(dim_head, kv_heads) if qk_norm else nn.Identity()

        self.attn = Attend(
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
    ) -> torch.Tensor:
        q_x = rearrange(self.to_q_x(x), "b n (h d) -> b h n d", h=self.heads)
        k_x = self.to_k_x(x)
        v_x = self.to_v_x(x)
        k_x, v_x = (rearrange(t, "b n (h d) -> b h n d", h=self.kv_heads) for t in (k_x, v_x))

        q_a = rearrange(self.to_q_a(a), "b n (h d) -> b h n d", h=self.heads)
        k_a = self.to_k_a(a)
        v_a = self.to_v_a(a)
        k_a, v_a = (rearrange(t, "b n (h d) -> b h n d", h=self.kv_heads) for t in (k_a, v_a))

        if self.qk_norm:
            q_x = self.q_x_norm(q_x)
            k_x = self.k_x_norm(k_x)
            q_a = self.q_a_norm(q_a)
            k_a = self.k_a_norm(k_a)

        # GQA
        k_x, v_x = (repeat(t, "b h n d -> b (r h) n d", r=self.heads // self.kv_heads) for t in (k_x, v_x))
        k_a, v_a = (repeat(t, "b h n d -> b (r h) n d", r=self.heads // self.kv_heads) for t in (k_a, v_a))

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
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 2,
        attn_qk_norm: bool = True,
        attn_causal: bool = True,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 8192,
        attn_infini: bool = True,
        attn_segment_len: int = 1024,
    ) -> None:
        super().__init__()
        # Modulation
        self.modulation_x = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_h, dim_h * 6, bias=True),
        )
        self.modulation_a = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_h, dim_h * 6, bias=True),
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

        self.attn = JointAttention(
            dim_h,
            attn_dim_head,
            attn_heads,
            attn_kv_heads,
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
    ) -> torch.Tensor:
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

    def forward(
        self: "MMDiTBlock",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, a, c, use_reentrant=True)
        else:
            return self.forward_body(x, a, c)


class FinalLayer(nn.Module):
    def __init__(self: "FinalLayer", dim_h: int, patch_size: int, dim_out: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_h, dim_h * 2, bias=True),
        )
        self.linear = nn.Linear(dim_h, patch_size * dim_out)

    def forward(self: "FinalLayer", x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class MMDiT(nn.Module):
    def __init__(
        self: "MMDiT",
        dim_in_x: int,
        dim_in_a: int,
        dim_in_c: int,
        dim_h: int,
        dim_h_mult: int = 4,
        patch_size: int = 4,
        depth: int = 12,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 2,
        attn_qk_norm: bool = True,
        attn_causal: bool = True,
        attn_use_rotary_emb: bool = True,
        attn_context_len: int = 8192,
        attn_infini: bool = True,
        attn_segment_len: int = 1024,
    ) -> None:
        super().__init__()

        self.dim_h = dim_h
        self.dim_in_x = dim_in_x
        self.patch_size = patch_size
        self.attn_context_len = (attn_context_len // patch_size) * 2  # We have two modalities
        self.attn_segment_len = (attn_segment_len // patch_size) * 2
        self.attn_infini = attn_infini

        self.emb_x = PatchEmbedding(dim_in_x, dim_h, patch_size)
        self.emb_a = PatchEmbedding(dim_in_a, dim_h, patch_size)

        self.feature_extractor_a = nn.Linear(dim_in_a * 2, dim_h)
        self.mlp_a = FeedForward(dim_h, dim_mult=dim_h_mult)
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
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_kv_heads=attn_kv_heads,
                    attn_qk_norm=attn_qk_norm,
                    attn_causal=attn_causal,
                    attn_use_rotary_emb=attn_use_rotary_emb,
                    attn_context_len=self.attn_context_len,
                    attn_infini=attn_infini,
                    attn_segment_len=self.attn_segment_len,
                )
                for _ in range(depth)
            ],
        )

        self.final_layer = FinalLayer(dim_h, self.patch_size, dim_h)
        self.out = nn.Linear(dim_h, dim_in_x)

        self.initialize_weights()

    def initialize_weights(self: "MMDiT") -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Basic initialization for layers
        self.apply(_basic_init)

        # Initialize embedder
        nn.init.normal_(self.mlp_a[0].weight, std=0.02)
        nn.init.normal_(self.mlp_a[2].weight, std=0.02)
        nn.init.normal_(self.mlp_time[1][0].weight, std=0.02)
        nn.init.normal_(self.mlp_time[1][2].weight, std=0.02)
        nn.init.normal_(self.mlp_cond[1][0].weight, std=0.02)
        nn.init.normal_(self.mlp_cond[1][2].weight, std=0.02)

        # Zero-out adaLN layers
        for block in self.blocks:
            nn.init.zeros_(block.modulation_x[1].weight)
            nn.init.zeros_(block.modulation_x[1].bias)
            nn.init.zeros_(block.modulation_a[1].weight)
            nn.init.zeros_(block.modulation_a[1].bias)

        # Zero-out output layer
        nn.init.zeros_(self.final_layer.modulation[1].weight)
        nn.init.zeros_(self.final_layer.modulation[1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def set_gradient_checkpointing(self: "MMDiT", value: bool) -> None:
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
        # Statistic audio features pooling
        mean_features = a.mean(dim=-1)
        std_features = a.std(dim=-1)
        h_a = torch.cat([mean_features, std_features], dim=1)
        h_a = self.feature_extractor_a(h_a)

        n = x.shape[-1]
        if self.attn_infini:
            # Pad to the closest multiple of attn_segment_len
            segment_len = self.attn_segment_len // 2  # We use half the segment length since we have two modalities
            segment_len *= self.patch_size  # times the patch size to get the real segment length
            pad_len = (segment_len - (n % segment_len)) % segment_len
        else:
            # Pad to the closest multiple of patch_size
            pad_len = (self.patch_size - (n % self.patch_size)) % self.patch_size

        x = F.pad(x, (0, pad_len), value=-1.0)
        a = F.pad(a, (0, pad_len), value=0.0)

        # Patchify the input
        x = self.emb_x(x)
        a = self.emb_a(a)

        # Add positional embedding and condition
        cond_mask = prob_mask_like((x.shape[0],), 1.0 - cond_drop_prob, device=x.device)
        cond_mask = rearrange(cond_mask, "b -> b 1")
        null_conds = repeat(self.null_cond, "d -> b d", b=x.shape[0])
        c = self.mlp_cond(c)
        c = torch.where(cond_mask, c, null_conds)

        c = c + self.mlp_time(t) + self.mlp_a(h_a)

        # Run the blocks
        for block in self.blocks:
            x, a = block(x, a, c)

        # Run the final layer
        x = self.final_layer(x, c)

        # Unpatchify the output
        x = rearrange(x, "b n (p d) -> b d (n p)", p=self.patch_size)
        return self.out(x)[:, :, :n]
