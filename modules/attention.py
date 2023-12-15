from collections import namedtuple
from typing import Optional

import torch
import torch.functional as F  # noqa: N812
import torch.nn as nn
from einops import rearrange
from packaging import version
from torch import einsum

_config = namedtuple("Config", ["enable_flash", "enable_math", "enable_mem_efficient"])


class Attention(nn.Module):
    def __init__(self: "Attention", dropout: float = 0.0, causal: bool = False, sdpa: bool = False) -> None:
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal

        assert not (sdpa and version.parse(torch.__version__) < version.parse("2.0.0")), "sdpa requires torch>=2.0.0"
        self.sdpa = sdpa
        self.register_buffer("mask", None, persistent=False)

        # sdpa configs
        self.cpu_config = _config(True, True, True)

        if not torch.cuda.is_available() or not self.sdpa:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major == 8 and device_properties.minor == 0:
            print("A100 detected, using flash attention")
            self.config = _config(True, False, False)
        else:
            print("Non-A100 detected, using math or memory efficient attention")
            self.config = _config(False, True, True)

    def sdpa_attn(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, heads, q_len, _, k_len, is_cuda, dtype = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.dtype,
        )

        k = rearrange(k, "b ... -> b 1 ...").expand_as(q)
        v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            mask = mask.expand(-1, heads, q_len, -1)
            if self.causal:
                causal_mask = torch.ones((q_len, k_len), device=q.device, dtype=torch.bool).triu(k_len - q_len + 1)
                mask = mask & ~causal_mask

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
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        return out.to(dtype) if config.enable_flash else out

    def forward(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.sdpa:
            assert attn_bias is None, "sdpa does not support attn_bias"
            return self.sdpa_attn(q, k, v, mask=mask)

        scale = q.shape[-1] ** -0.5
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        if attn_bias is not None:
            sim += attn_bias

        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device=sim.device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self: "MultiHeadAttention",
        dim: int,
        context_dim: Optional[int] = None,
        dim_head: int = 64,
        heads: int = 8,
        causal: bool = False,
        dropout: float = 0.25,
        sdpa: bool = False,
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.is_cross_attention = is_cross_attention

        inner_dim = dim_head * heads

        if is_cross_attention:
            assert context_dim is not None, "context_dim must be provided for cross attention"
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attention = Attention(dropout=dropout, causal=causal, sdpa=sdpa)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self: "MultiHeadAttention",
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_cross_attention:
            assert context is not None, "context must be provided for cross attention"
            q = self.to_q(x)
            k, v = self.to_kv(context).chunk(2, dim=-1)
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        out = self.attention(q, k, v, mask=mask, attn_bias=attn_bias)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out(out)
        return out
