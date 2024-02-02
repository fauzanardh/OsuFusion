from collections import namedtuple
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from packaging import version
from torch.nn import functional as F  # noqa: N812

from osu_fusion.modules.utils import apply_rotary_pos_emb

_config = namedtuple("Config", ["enable_flash", "enable_math", "enable_mem_efficient"])


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self: "RotaryPositionEmbedding",
        dim: int,
        theta: int = 10000,
        scale_base: int = 8192,
    ) -> None:
        super().__init__()
        self.scale_base = scale_base

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self: "RotaryPositionEmbedding", x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-1]
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seq_len
            t = torch.arange(
                seq_len,
                dtype=torch.float32,
                device=x.device,
            )
            t *= self.scale_base / seq_len
            freqs = torch.einsum("i , j -> i j", self.inv_freq.to(x.dtype), t)
            emb = torch.cat([freqs, freqs], dim=-2)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self: "RotaryPositionEmbedding", q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(q)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class Attention(nn.Module):
    def __init__(self: "Attention", sdpa: bool = False, linear: bool = False) -> None:
        super().__init__()
        assert not (sdpa and version.parse(torch.__version__) < version.parse("2.0.0")), "sdpa requires torch>=2.0.0"
        assert not (sdpa and linear), "sdpa can't be used with linear attention"
        self.sdpa = sdpa
        self.linear = linear

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
            if config.enable_flash and any(
                [q.dtype == torch.float64, k.dtype == torch.float64, v.dtype == torch.float64],
            ):
                q = q.half()
                k = k.half()
                v = v.half()
            scale = q.shape[-2] ** -0.5
            q, k, v = (rearrange(t, "b h d n -> b n h d") for t in (q, k, v))
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
            )
            out = rearrange(out, "b n h d -> b h d n")

        return out.to(dtype) if config.enable_flash else out

    def attn_linear(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        scale = q.shape[-1] ** -0.5
        q = q.softmax(dim=-2) * scale
        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        return out

    def attn(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        scale = q.shape[-2] ** -0.5
        q = q * scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h d i", attn, v)
        return out

    def forward(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if self.sdpa:
            return self.sdpa_attn(q, k, v)
        elif self.linear:
            return self.attn_linear(q, k, v)
        else:
            return self.attn(q, k, v)


class MultiHeadAttention(nn.Module):
    def __init__(
        self: "MultiHeadAttention",
        dim: int,
        dim_context: Optional[int] = None,
        dim_head: int = 32,
        heads: int = 8,
        sdpa: bool = False,
        linear: bool = False,
        use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads

        inner_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)
        self.rotary_emb = RotaryPositionEmbedding(dim_head) if use_rotary_emb else None
        self.attention = Attention(sdpa=sdpa, linear=linear)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(
        self: "MultiHeadAttention",
        x: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=-2)
        q, k, v = (rearrange(t, "b (h d) n -> b h d n", h=self.heads) for t in (q, k, v))
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)

        q, k, v = (t.contiguous() for t in (q, k, v))
        out = self.attention(q, k, v)
        out = rearrange(out, "b h d n -> b (h d) n")

        out = self.to_out(out)
        return out
