from collections import namedtuple
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from packaging import version
from torch.nn import functional as F  # noqa: N812

from osu_fusion.modules.utils import apply_rotary_pos_emb

_config = namedtuple("Config", ["enable_flash", "enable_math", "enable_mem_efficient"])


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self: "RotaryPositionEmbedding",
        dim: int,
        theta: int = 10000,
        interpolation_factor: float = 1.0,
        scale_base: int = 16384,
    ) -> None:
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)
        self.interpolation_factor = interpolation_factor

        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer("scale", scale, persistent=False)

        self.register_buffer("cached_freqs", None, persistent=False)
        self.register_buffer("cached_scale", None, persistent=False)

    def get_seq_pos(
        self: "RotaryPositionEmbedding",
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> torch.Tensor:
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolation_factor

    def get_scale(
        self: "RotaryPositionEmbedding",
        t: torch.Tensor,
        seq_len: Optional[int] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        should_cache = seq_len is not None

        if should_cache and self.cached_scale is not None and (offset + seq_len) <= self.cached_scale.shape[0]:
            return self.cached_scale[offset : (offset + seq_len)]

        power = (t - t.shape[-1] // 2) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)

        if should_cache:
            self.register_buffer("cached_scale", scale, persistent=False)

        return scale

    def rotate_queries_and_keys(
        self: "RotaryPositionEmbedding",
        q: torch.Tensor,
        k: torch.Tensor,
        seq_dim: int = -2,
    ) -> torch.Tensor:
        device, dtype, q_seq_len, k_seq_len = q.device, q.dtype, q.shape[seq_dim], k.shape[seq_dim]
        assert q_seq_len == k_seq_len, "sequence lengths of queries and keys must match"

        seq = self.get_seq_pos(q_seq_len, device, dtype)
        freqs = self(seq, seq_len=q_seq_len)
        scale = self.get_scale(seq, seq_len=q_seq_len).to(dtype)

        rotated_q = apply_rotary_pos_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_pos_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def forward(
        self: "RotaryPositionEmbedding",
        t: torch.Tensor,
        seq_len: Optional[int] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        should_cache = seq_len is not None

        if should_cache and self.cached_freqs is not None and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset : (offset + seq_len)]

        freqs = self.freqs

        freqs = torch.einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if should_cache:
            self.register_buffer("cached_freqs", freqs, persistent=False)

        return freqs


class Attention(nn.Module):
    def __init__(self: "Attention", dropout: float = 0.0, sdpa: bool = False) -> None:
        super().__init__()
        self.dropout = dropout

        assert not (sdpa and version.parse(torch.__version__) < version.parse("2.0.0")), "sdpa requires torch>=2.0.0"
        self.sdpa = sdpa

        # sdpa configs
        self.cpu_config = _config(True, True, True)

        if not torch.cuda.is_available() or not self.sdpa:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = _config(True, False, False)
        else:
            self.cuda_config = _config(False, True, True)

    def sdpa_attn(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        is_cuda, dtype = q.is_cuda, q.dtype
        config = self.cuda_config if is_cuda else self.cpu_config

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            if config.enable_flash:
                # convert to half precision
                q = q.half()
                k = k.half()
                v = v.half()

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
            )

        return out.to(dtype) if config.enable_flash else out

    def forward(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if self.sdpa:
            return self.sdpa_attn(q, k, v)

        scale = q.shape[-1] ** -0.5
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale

        attn = sim.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self: "MultiHeadAttention",
        dim: int,
        dim_context: Optional[int] = None,
        dim_head: int = 32,
        heads: int = 8,
        dropout: float = 0.1,
        sdpa: bool = False,
        is_cross_attention: bool = False,
        use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.is_cross_attention = is_cross_attention

        inner_dim = dim_head * heads

        if is_cross_attention:
            assert dim_context is not None, "context_dim must be provided for cross attention"
            self.to_q = nn.Linear(dim, inner_dim)
            self.to_kv = nn.Linear(dim_context, inner_dim * 2)
            self.rotary_emb = None
        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3)
            self.rotary_emb = RotaryPositionEmbedding(dim_head) if use_rotary_emb else None
        self.attention = Attention(dropout=dropout, sdpa=sdpa)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self: "MultiHeadAttention",
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_cross_attention:
            assert context is not None, "context must be provided for cross attention"
            q = self.to_q(x)
            k, v = self.to_kv(context).chunk(2, dim=-1)
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))

        if self.rotary_emb is not None:
            q, k = self.rotary_emb.rotate_queries_and_keys(q, k)

        out = self.attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out(out)
        return out
