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
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self: "RotaryPositionEmbedding", x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seq_len
            t = torch.arange(
                x.shape[-2],
                dtype=torch.float32,
                device=x.device,
            )
            freqs = torch.einsum("i , j -> i j", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self: "RotaryPositionEmbedding", q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(q)

        return apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached), apply_rotary_pos_emb(
            k,
            self._cos_cached,
            self._sin_cached,
        )


class Attention(nn.Module):
    def __init__(self: "Attention", dropout: float = 0.0, sdpa: bool = False, linear: bool = False) -> None:
        super().__init__()
        self.dropout = dropout

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

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
            )

        return out.to(dtype) if config.enable_flash else out

    def attn_linear(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        context = torch.einsum("b h n d, b h n e -> b h d e", k, v)
        out = torch.einsum("b h d e, b h n d -> b h n e", context, q)
        return out

    def attn(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        attn = sim.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        return out

    def forward(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if scale is None:
            scale = q.shape[-1] ** -0.5
        q = q * scale

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
        dropout: float = 0.1,
        sdpa: bool = False,
        linear: bool = False,
        is_cross_attention: bool = False,
        use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        self.is_cross_attention = is_cross_attention

        inner_dim = dim_head * heads
        if is_cross_attention:
            assert dim_context is not None, "context_dim must be provided for cross attention"
            self.to_q = nn.Conv1d(dim, inner_dim, 1, bias=False)
            self.to_kv = nn.Conv1d(dim_context, inner_dim * 2, 1, bias=False)
            self.rotary_emb = None
        else:
            self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)
            self.rotary_emb = RotaryPositionEmbedding(dim_head) if use_rotary_emb else None
        self.attention = Attention(dropout=dropout, sdpa=sdpa, linear=linear)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(
        self: "MultiHeadAttention",
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_cross_attention:
            assert context is not None, "context must be provided for cross attention"
            q = self.to_q(x)
            k, v = self.to_kv(context).chunk(2, dim=-2)
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-2)

        q, k, v = (rearrange(t, "b (h d) n -> b h n d", h=self.heads) for t in (q, k, v))
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)

        q, k, v = (t.contiguous() for t in (q, k, v))

        out = self.attention(q, k, v, scale=self.scale)
        out = rearrange(out, "b h n d -> b (h d) n")

        out = self.to_out(out)
        return out
