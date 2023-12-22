from collections import namedtuple
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from packaging import version
from torch.nn import functional as F  # noqa: N812

_config = namedtuple("Config", ["enable_flash", "enable_math", "enable_mem_efficient"])


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int = 0,
    scale: float = 1.0,
    seq_dim: int = -2,
) -> torch.Tensor:
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"Features dim {t.shape[-1]} must be larger than the positional encoding dim {rot_dim}"

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self: "RotaryPositionEmbedding",
        dim: int,
        theta: int = 10000,
        interpolation_factor: float = 1.0,
        scale_base: int = 8192,
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

        q_seq = self.get_seq_pos(q_seq_len, device, dtype)
        q_freqs = self(q_seq, seq_len=q_seq_len)
        q_scale = self.get_scale(q_seq, seq_len=q_seq_len).to(dtype)

        k_seq = self.get_seq_pos(k_seq_len, device, dtype)
        k_freqs = self(k_seq, seq_len=k_seq_len)
        k_scale = self.get_scale(k_seq, seq_len=k_seq_len).to(dtype)

        rotated_q = apply_rotary_pos_emb(q_freqs, q, scale=q_scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_pos_emb(k_freqs, k, scale=k_scale**-1, seq_dim=seq_dim)

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


class DynamicPositionBias(nn.Module):
    def __init__(
        self: "DynamicPositionBias",
        dim: int,
        heads: int = 8,
        depth: int = 2,
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
        max_seq_len = max(i, j)
        device = self.device
        seq_arange = torch.arange(i, device=device)
        context_arange = torch.arange(j, device=device)
        indices = rearrange(seq_arange, "i -> i 1") - rearrange(context_arange, "j -> 1 j")
        indices += max_seq_len - 1

        pos = torch.arange(-max_seq_len + 1, max_seq_len, device=device).float()
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
        use_rotary_emb: bool = True,
        use_dynamic_position_bias: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.is_cross_attention = is_cross_attention

        inner_dim = dim_head * heads

        self.rel_pos = DynamicPositionBias(dim // 4, heads=heads) if use_dynamic_position_bias else None
        self.rotary_emb = RotaryPositionEmbedding(dim_head) if use_rotary_emb else None
        if is_cross_attention:
            assert dim_context is not None, "context_dim must be provided for cross attention"
            self.to_q = nn.Linear(dim, inner_dim)
            self.to_kv = nn.Linear(dim_context, inner_dim * 2)
        else:
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
        if self.rotary_emb is not None:
            q, k = self.rotary_emb.rotate_queries_and_keys(q, k)

        attn_bias = None
        if self.rel_pos is not None:
            i, j = q.shape[-2], k.shape[-2]
            attn_bias = self.rel_pos(i, j)

        out = self.attention(q, k, v, attn_bias=attn_bias)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out(out)
        return out
