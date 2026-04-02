import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, pack, unpack
from torch.nn import functional as F
from torch.profiler import record_function

try:
    from xformers.ops import memory_efficient_attention

    print("Using xformers memory-efficient attention")
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

from osu_fusion.modules.norms import MultiHeadRMSNorm
from osu_fusion.modules.utils import dummy_context_manager

DEBUG = os.environ.get("DEBUG", False)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.amp.autocast("cuda", dtype=torch.float32)
def apply_rotary_pos_emb(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rot_dim = cos.shape[-1]
    orig_dtype = t.dtype

    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]

    scale = 1.0 if scale is None else scale
    t = (t * cos * scale) + (rotate_half(t) * sin * scale)
    t = torch.cat([t, t_unrotated], dim=-1)
    return t.to(orig_dtype)


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self: "RotaryPositionEmbedding",
        dim: int,
        scale_base: int = 4096,
        theta: int = 10000,
        theta_rescale_factor: float = 4.0,
        interpolation_factor: float = 1.0,
        use_xpos: bool = False,
    ) -> None:
        super().__init__()
        assert interpolation_factor >= 1.0, "Interpolation factor must be >= 1.0"

        self.dim = dim
        self.scale_base = scale_base
        self.interpolation_factor = interpolation_factor
        self.use_xpos = use_xpos

        theta *= theta_rescale_factor ** (dim / (dim - 2))
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        if use_xpos:
            scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
            self.register_buffer("scale", scale, persistent=False)
        else:
            self.register_buffer("scale", None, persistent=False)

        self._cached_cos: Optional[torch.Tensor] = None
        self._cached_sin: Optional[torch.Tensor] = None
        self._cached_scale: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0
        self._cached_device: Optional[torch.device] = None

    def _compute_scale(
        self: "RotaryPositionEmbedding",
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.use_xpos:
            return None

        t = torch.arange(seq_len, device=device, dtype=dtype)
        max_pos = t.max() + 1
        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale.to(dtype) ** rearrange(power, "n -> n 1")
        scale = torch.stack([scale, scale], dim=-1)
        return rearrange(scale, "... d r -> ... (d r)")

    @torch.amp.autocast("cuda", dtype=torch.float32)
    def _get_cos_sin_scale(
        self: "RotaryPositionEmbedding",
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        seq_len = x.shape[-2]
        device = x.device

        if seq_len > self._cached_seq_len or self._cached_device != device:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.einsum("i, j -> i j", t, self.inv_freq.to(torch.float32)) / self.interpolation_factor
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cached_cos = rearrange(emb.cos(), "n d -> 1 1 n d")
            self._cached_sin = rearrange(emb.sin(), "n d -> 1 1 n d")
            self._cached_seq_len = seq_len
            self._cached_device = device

            scale = self._compute_scale(seq_len, device, torch.float32)
            if scale is not None:
                scale = rearrange(scale, "n d -> 1 1 n d")
            self._cached_scale = scale

        cos = self._cached_cos[:, :, :seq_len, :]
        sin = self._cached_sin[:, :, :seq_len, :]
        scale = self._cached_scale[:, :, :seq_len, :] if self._cached_scale is not None else None

        return cos, sin, scale

    def forward(
        self: "RotaryPositionEmbedding",
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin, scale = self._get_cos_sin_scale(q)

        return (
            apply_rotary_pos_emb(q, cos, sin, scale),
            apply_rotary_pos_emb(k, cos, sin, scale),
        )

    def forward_single(
        self: "RotaryPositionEmbedding",
        x: torch.Tensor,
    ) -> torch.Tensor:
        cos, sin, scale = self._get_cos_sin_scale(x)
        return apply_rotary_pos_emb(x, cos, sin, scale)


class Attend(nn.Module):
    def __init__(self: "Attend") -> None:
        super().__init__()
        self.use_xformers = XFORMERS_AVAILABLE
        if not torch.cuda.is_available():
            self.use_xformers = False
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major >= 8 and device_properties.minor >= 0:
            self.can_use_bf16 = True
        else:
            self.can_use_bf16 = False

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self: "Attend",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dtype = v.dtype

        qkv_dtype = torch.bfloat16 if self.can_use_bf16 else torch.float16
        q = q.to(qkv_dtype)
        k = k.to(qkv_dtype)
        v = v.to(qkv_dtype)

        if self.use_xformers and attn_mask is None:
            # xformers fast path: no mask (inference)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            out = memory_efficient_attention(q, k, v)
            out = out.transpose(1, 2)
        else:
            attn_mask = attn_mask.to(qkv_dtype) if attn_mask is not None else None
            q, k, v = (t.contiguous() for t in (q, k, v))
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
            )

        return out.to(dtype)


class Attention(nn.Module):
    def __init__(
        self: "Attention",
        dim_in: int,
        dim_head: int,
        heads: int,
        rotary_emb: Optional[RotaryPositionEmbedding] = None,
        context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.heads = heads

        self.to_qkv = nn.Linear(dim_in, dim_head * heads * 3, bias=False)
        self.rotary_emb = (
            rotary_emb if rotary_emb is not None else RotaryPositionEmbedding(dim_head, scale_base=context_len)
        )
        self.q_norm = MultiHeadRMSNorm(dim_head, heads)
        self.k_norm = MultiHeadRMSNorm(dim_head, heads)

        self.attn = Attend()
        self.to_out = nn.Linear(dim_head * heads, dim_in)

    def forward_body(
        self: "Attention",
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))

        q, k = self.rotary_emb(q, k)
        q = self.q_norm(q)
        k = self.k_norm(k)

        out = self.attn(q, k, v, attn_mask=attn_mask)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    def forward(
        self: "Attention",
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("Attention")
        with context_manager:
            return self.forward_body(x, attn_mask=attn_mask)


class CrossAttention(nn.Module):
    def __init__(
        self: "CrossAttention",
        dim_in: int,
        dim_head: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.heads = heads

        self.to_q = nn.Linear(dim_in, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim_in, dim_head * heads * 2, bias=False)
        self.q_norm = MultiHeadRMSNorm(dim_head, heads)
        self.k_norm = MultiHeadRMSNorm(dim_head, heads)

        self.attn = Attend()
        self.to_out = nn.Linear(dim_head * heads, dim_in)

    def forward_body(
        self: "CrossAttention",
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)

        k, v = self.to_kv(context).chunk(2, dim=-1)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (k, v))

        q = self.q_norm(q)
        k = self.k_norm(k)

        out = self.attn(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    def forward(self: "CrossAttention", x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("CrossAttention")
        with context_manager:
            return self.forward_body(x, context)


class JointAttention(nn.Module):
    def __init__(
        self: "JointAttention",
        dim_in: int,
        dim_head: int,
        heads: int,
        rotary_emb: Optional[RotaryPositionEmbedding] = None,
        context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.rotary_emb = (
            rotary_emb if rotary_emb is not None else RotaryPositionEmbedding(dim_head, scale_base=context_len)
        )

        self.to_qkv_x = nn.Linear(dim_in, dim_head * heads * 3, bias=False)
        self.to_qkv_a = nn.Linear(dim_in, dim_head * heads * 3, bias=False)

        self.q_norm_x = MultiHeadRMSNorm(dim_head, heads)
        self.k_norm_x = MultiHeadRMSNorm(dim_head, heads)

        self.q_norm_a = MultiHeadRMSNorm(dim_head, heads)
        self.k_norm_a = MultiHeadRMSNorm(dim_head, heads)

        self.attn = Attend()
        self.to_out_x = nn.Linear(dim_head * heads, dim_in)
        self.to_out_a = nn.Linear(dim_head * heads, dim_in)

    def forward_body(
        self: "JointAttention",
        x: torch.Tensor,
        a: torch.Tensor,
        mask_x: Optional[torch.Tensor] = None,
        mask_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_x, k_x, v_x = self.to_qkv_x(x).chunk(3, dim=-1)
        q_x, k_x, v_x = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q_x, k_x, v_x))

        q_a, k_a, v_a = self.to_qkv_a(a).chunk(3, dim=-1)
        q_a, k_a, v_a = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q_a, k_a, v_a))

        if self.rotary_emb is not None:
            q_x, k_x = self.rotary_emb(q_x, k_x)
            q_a, k_a = self.rotary_emb(q_a, k_a)

        q_x, k_x = self.q_norm_x(q_x), self.k_norm_x(k_x)
        q_a, k_a = self.q_norm_a(q_a), self.k_norm_a(k_a)

        q, seq_shape = pack([q_a, q_x], "b h * d")
        k, _ = pack([k_a, k_x], "b h * d")
        v, _ = pack([v_a, v_x], "b h * d")

        attn_mask = None
        if mask_x is not None or mask_a is not None:
            n_a = a.shape[1]
            n_x = x.shape[1]
            device = x.device
            if mask_a is None:
                mask_a = torch.ones(x.shape[0], n_a, device=device, dtype=torch.bool)
            if mask_x is None:
                mask_x = torch.ones(x.shape[0], n_x, device=device, dtype=torch.bool)
            combined_mask = torch.cat([mask_a, mask_x], dim=1)
            attn_mask = torch.zeros(combined_mask.shape, device=device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~combined_mask, float("-inf"))
            attn_mask = attn_mask[:, None, None, :]

        out = self.attn(q, k, v, attn_mask=attn_mask)
        out_a, out_x = unpack(out, seq_shape, "b h * d")
        out_x, out_a = (rearrange(t, "b h n d -> b n (h d)") for t in (out_x, out_a))
        return self.to_out_x(out_x), self.to_out_a(out_a)

    def forward(
        self: "JointAttention",
        x: torch.Tensor,
        a: torch.Tensor,
        mask_x: Optional[torch.Tensor] = None,
        mask_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("JointAttention")
        with context_manager:
            return self.forward_body(x, a, mask_x=mask_x, mask_a=mask_a)
