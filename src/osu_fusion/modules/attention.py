import os
from typing import Optional

import torch
import torch.nn as nn
from einops import pack, rearrange, repeat, unpack
from packaging import version
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import record_function

from osu_fusion.modules.norms import RMSNorm
from osu_fusion.modules.utils import dummy_context_manager

DEBUG = os.environ.get("DEBUG", False)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self: "RotaryPositionEmbedding",
        dim: int,
        theta: int = 10000,
        scale_base: int = 4096,
    ) -> None:
        super().__init__()
        self.scale_base = scale_base

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    @torch.amp.autocast("cuda", dtype=torch.float32)
    def _update_cos_sin_tables(self: "RotaryPositionEmbedding", x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seq_len
            t = torch.arange(
                seq_len,
                dtype=x.dtype,
                device=x.device,
            )
            t *= self.scale_base / seq_len
            freqs = torch.einsum("i , j -> i j", t, self.inv_freq.to(x.dtype))
            emb = torch.cat([freqs, freqs], dim=-1)

            self._cos_cached = rearrange(emb.cos(), "n d -> 1 1 n d")
            self._sin_cached = rearrange(emb.sin(), "n d -> 1 1 n d")

        return self._cos_cached, self._sin_cached

    def forward(self: "RotaryPositionEmbedding", q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(q)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class Attend(nn.Module):
    def __init__(self: "Attend") -> None:
        super().__init__()
        assert not version.parse(torch.__version__) < version.parse("2.0.0"), "sdpa requires torch>=2.0.0"
        self.can_use_bf16 = True
        self.cpu_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

        if not torch.cuda.is_available():
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major >= 8 and device_properties.minor >= 0:
            self.cuda_backends = [SDPBackend.FLASH_ATTENTION]
        else:
            self.can_use_bf16 = False
            self.cuda_backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self: "Attend",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        is_cuda, dtype = v.is_cuda, v.dtype
        backends = self.cuda_backends if is_cuda else self.cpu_backends

        qkv_dtype = torch.bfloat16 if self.can_use_bf16 else torch.float16
        with sdpa_kernel(backends):
            q = q.to(qkv_dtype)
            k = k.to(qkv_dtype)
            v = v.to(qkv_dtype)
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
        kv_heads: int,
        context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.kv_heads = kv_heads

        self.norm = RMSNorm(dim_in)
        self.to_q = nn.Linear(dim_in, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim_in, dim_head * kv_heads * 2, bias=False)
        self.rotary_emb = RotaryPositionEmbedding(dim_head, scale_base=context_len)

        self.attn = Attend()
        self.to_out = nn.Linear(dim_head * heads, dim_in)

    def forward_body(self: "Attention", x: torch.Tensor) -> torch.Tensor:
        # Pre-norm
        x = self.norm(x)

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)

        k, v = self.to_kv(x).chunk(2, dim=-1)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.kv_heads) for t in (k, v))

        q, k = self.rotary_emb(q, k)
        q = q * self.scale

        # GQA
        k, v = (repeat(t, "b h n d -> b (r h) n d", r=self.heads // self.kv_heads) for t in (k, v))

        out = self.attn(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    def forward(self: "Attention", x: torch.Tensor) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("Attention")
        with context_manager:
            return self.forward_body(x)


class JointAttention(nn.Module):
    def __init__(
        self: "JointAttention",
        dim_in: int,
        dim_head: int,
        heads: int,
        kv_heads: int,
        context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.kv_heads = kv_heads

        self.norm_x = RMSNorm(dim_in)
        self.to_q_x = nn.Linear(dim_in, dim_head * heads, bias=False)
        self.to_kv_x = nn.Linear(dim_in, dim_head * kv_heads * 2, bias=False)

        self.norm_a = RMSNorm(dim_in)
        self.to_q_a = nn.Linear(dim_in, dim_head * heads, bias=False)
        self.to_kv_a = nn.Linear(dim_in, dim_head * kv_heads * 2, bias=False)

        self.rotary_emb = RotaryPositionEmbedding(dim_head, scale_base=context_len)
        self.attn = Attend()
        self.to_out_x = nn.Linear(dim_head * heads, dim_in)
        self.to_out_a = nn.Linear(dim_head * heads, dim_in)

    def forward_body(
        self: "JointAttention",
        x: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm
        x = self.norm_x(x)
        a = self.norm_a(a)

        q_x = rearrange(self.to_q_x(x), "b n (h d) -> b h n d", h=self.heads)
        k_x, v_x = self.to_kv_x(x).chunk(2, dim=-1)
        k_x, v_x = (rearrange(t, "b n (h d) -> b h n d", h=self.kv_heads) for t in (k_x, v_x))

        q_a = rearrange(self.to_q_a(a), "b n (h d) -> b h n d", h=self.heads)
        k_a, v_a = self.to_kv_a(a).chunk(2, dim=-1)
        k_a, v_a = (rearrange(t, "b n (h d) -> b h n d", h=self.kv_heads) for t in (k_a, v_a))

        q_x, k_x = self.rotary_emb(q_x, k_x)
        q_a, k_a = self.rotary_emb(q_a, k_a)

        q_x = q_x * self.scale
        q_a = q_a * self.scale

        # GQA
        k_x, v_x = (repeat(t, "b h n d -> b (r h) n d", r=self.heads // self.kv_heads) for t in (k_x, v_x))
        k_a, v_a = (repeat(t, "b h n d -> b (r h) n d", r=self.heads // self.kv_heads) for t in (k_a, v_a))

        q, seq_shape = pack([q_a, q_x], "b h * d")
        k, _ = pack([k_a, k_x], "b h * d")
        v, _ = pack([v_a, v_x], "b h * d")

        out = self.attn(q, k, v)
        out_a, out_x = unpack(out, seq_shape, "b h * d")
        out_x, out_a = (rearrange(t, "b h n d -> b n (h d)") for t in (out_x, out_a))
        return self.to_out_x(out_x), self.to_out_a(out_a)

    def forward(self: "JointAttention", x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("JointAttention")
        with context_manager:
            return self.forward_body(x, a)
