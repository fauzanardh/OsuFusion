from collections import namedtuple
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from packaging import version
from torch.nn import functional as F  # noqa: N812

_config = namedtuple("Config", ["enable_flash", "enable_math", "enable_mem_efficient"])


class DynamicPositionBias(nn.Module):
    def __init__(
        self: "DynamicPositionBias",
        dim: int,
        heads: int = 8,
        depth: int = 4,
        log_distance: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.log_distance = log_distance

        self.layers = [
            nn.Sequential(
                nn.Linear(1, dim),
                nn.LayerNorm(dim) if normalize else nn.Identity(),
                nn.SiLU(),
            ),
        ]

        for _ in range(depth - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim) if normalize else nn.Identity(),
                    nn.SiLU(),
                ),
            )

        self.layers.append(nn.Linear(dim, heads))
        self.layers = nn.ModuleList(self.layers)

    @property
    def device(self: "DynamicPositionBias") -> torch.device:
        return next(self.parameters()).device

    def forward(self: "DynamicPositionBias", i: int, j: int) -> torch.Tensor:
        assert i == j, "i and j must be equal for DynamicPositionBias"
        n, device = j, self.device

        seq_arange = torch.arange(n, device=device)
        context_arange = torch.arange(n, device=device)
        indices = rearrange(seq_arange, "i -> i 1") - rearrange(context_arange, "j -> 1 j")
        indices += n - 1

        pos = torch.arange(-n + 1, n, device=device).float()
        pos = rearrange(pos, "... -> ... 1")

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)

        for layer in self.layers:
            pos = layer(pos)

        bias = pos[indices]
        bias = rearrange(bias, "i j h -> h i j")
        return bias


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
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        is_cuda, dtype = q.is_cuda, q.dtype
        config = self.cuda_config if is_cuda else self.cpu_config

        attn_bias = rearrange(attn_bias, "h i j -> 1 h i j") if attn_bias is not None else None

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
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
            )

        return out.to(dtype) if config.enable_flash else out

    def forward(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.sdpa:
            return self.sdpa_attn(q, k, v)

        scale = q.shape[-1] ** -0.5
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        sim = sim + attn_bias if attn_bias is not None else sim

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
    ) -> None:
        super().__init__()
        self.heads = heads
        self.is_cross_attention = is_cross_attention

        inner_dim = dim_head * heads

        if is_cross_attention:
            assert dim_context is not None, "context_dim must be provided for cross attention"
            self.rel_pos = None
            self.to_q = nn.Linear(dim, inner_dim)
            self.to_kv = nn.Linear(dim_context, inner_dim * 2)
        else:
            self.rel_pos = DynamicPositionBias(dim // 4, heads=heads)
            self.to_qkv = nn.Linear(dim, inner_dim * 3)
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

        attn_bias = None
        if self.rel_pos is not None:
            i, j = q.shape[-2], k.shape[-2]
            attn_bias = self.rel_pos(i, j)

        out = self.attention(q, k, v, attn_bias=attn_bias)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out(out)
        return out
