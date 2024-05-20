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
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self: "RotaryPositionEmbedding", x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seq_len
            t = torch.arange(
                seq_len,
                dtype=torch.float32,
                device=x.device,
            )
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


class Attention(nn.Module):
    def __init__(
        self: "Attention",
        dim_head: int,
        heads: int = 8,
        causal: bool = True,
        use_rotary_emb: bool = True,
        infini: bool = True,
        segment_len: int = 1024,
    ) -> None:
        super().__init__()
        assert not version.parse(torch.__version__) < version.parse("2.0.0"), "sdpa requires torch>=2.0.0"
        self.heads = heads
        self.use_rotary_emb = use_rotary_emb
        self.causal = causal
        self.infini = infini
        self.segment_len = segment_len

        # sdpa configs
        self.cpu_config = _config(True, True, True)

        if use_rotary_emb:
            self.rotary_emb = RotaryPositionEmbedding(dim_head)

        if infini:
            self.gate = nn.Parameter(torch.full((1, heads, 1, 1), -100.0))

        if not torch.cuda.is_available():
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = _config(True, False, False)
        else:
            self.cuda_config = _config(False, True, True)

    def forward_sdpa(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        is_cuda, dtype = v.is_cuda, v.dtype
        config = self.cuda_config if is_cuda else self.cpu_config

        if self.use_rotary_emb:
            q, k = self.rotary_emb(q, k)

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            q = q.half()
            k = k.half()
            v = v.half()
            q, k, v = (t.contiguous() for t in (q, k, v))
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=self.causal,
            )

        return out.to(dtype)

    def _retrieve_from_memory(
        self: "Attention",
        q: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        norm_term: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if memory is None or norm_term is None:
            return torch.zeros_like(q)

        q = F.elu(q) + 1.0

        memory = torch.matmul(q, memory)
        norm_term = torch.matmul(
            q,
            rearrange(norm_term, "b h 1 d -> b h d 1"),
        )

        return memory / (norm_term + 1e-6)

    def _update_memory(
        self: "Attention",
        k: torch.Tensor,
        v: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        norm_term: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k = F.elu(k) + 1.0

        if memory is not None:
            memory = memory + torch.matmul(rearrange(k, "b h n d -> b h d n"), v)
        else:
            memory = torch.matmul(rearrange(k, "b h n d -> b h d n"), v)

        if norm_term is not None:  # noqa: SIM108
            norm_term = norm_term + k.sum(dim=-2, keepdim=True)
        else:
            norm_term = k.sum(dim=-2, keepdim=True)

        return memory, norm_term

    def forward_infini(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        n_segments = q.shape[-2] // self.segment_len  # Assume sequence length is divisible by segment length
        q, k, v = (rearrange(t, "b h (s n) d -> b h s n d", s=n_segments) for t in (q, k, v))

        outputs = []
        memory = None
        norm_term = None
        for idx in range(n_segments):
            q_segment = q[:, :, idx, :, :]
            k_segment = k[:, :, idx, :, :]
            v_segment = v[:, :, idx, :, :]

            memory_output = self._retrieve_from_memory(q_segment, memory, norm_term)
            updated_memory, updated_norm_term = self._update_memory(
                k_segment,
                v_segment,
                memory,
                norm_term,
            )
            memory = updated_memory.detach()
            norm_term = updated_norm_term.detach()

            attn = self.forward_sdpa(q_segment, k_segment, v_segment)
            combined_output = (F.sigmoid(self.gate) * memory_output) + (1 - F.sigmoid(self.gate)) * attn
            outputs.append(combined_output)

        out = torch.cat(outputs, dim=-2)
        return out

    def forward(
        self: "Attention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if self.infini:
            with torch.autocast(device_type=q.device.type, dtype=torch.float32):
                return self.forward_infini(q, k, v)
        return self.forward_sdpa(q, k, v)


class MultiHeadAttention(nn.Module):
    def __init__(
        self: "MultiHeadAttention",
        dim: int,
        dim_head: int = 32,
        heads: int = 8,
        causal: bool = True,
        use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads

        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, 1, bias=False)
        self.rotary_emb = RotaryPositionEmbedding(dim_head) if use_rotary_emb else None
        self.attention = Attention(causal=causal)
        self.to_out = nn.Linear(inner_dim, dim, 1)

    def forward(
        self: "MultiHeadAttention",
        x: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=-2)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)

        out = self.attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out(out)
        return out
