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
    def __init__(
        self: "Attend",
        dim_head: int,
        heads: int = 8,
        causal: bool = False,
        use_rotary_emb: bool = True,
        context_len: int = 8192,
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
            self.rotary_emb = RotaryPositionEmbedding(dim_head, scale_base=context_len)

        if infini:
            self.gate = nn.Parameter(torch.full((1, heads, 1, 1), 0.0))  # Start at 50% memory

        if not torch.cuda.is_available():
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major >= 8 and device_properties.minor >= 0:
            self.cuda_config = _config(True, False, False)
        else:
            self.cuda_config = _config(False, True, True)

    def forward_sdpa(
        self: "Attend",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        is_cuda, dtype = v.is_cuda, v.dtype
        config = self.cuda_config if is_cuda else self.cpu_config

        qkv_dtype = torch.bfloat16 if self.cuda_config.enable_flash else torch.float16
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
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

    def _retrieve_from_memory(
        self: "Attend",
        q: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        norm_term: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.autocast(device_type=q.device.type, dtype=torch.float32):
            if memory is None or norm_term is None:
                return torch.zeros_like(q, device=q.device)

            q = F.elu(q) + 1.0

            memory = torch.matmul(q, memory)
            norm_term = torch.matmul(
                q,
                rearrange(norm_term, "b h 1 d -> b h d 1"),
            )

            return memory / (norm_term + 1e-6)

    def _update_memory(
        self: "Attend",
        k: torch.Tensor,
        v: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        norm_term: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.autocast(device_type=k.device.type, dtype=torch.float32):
            k = F.elu(k) + 1.0

            if memory is not None:
                memory = memory + torch.matmul(rearrange(k, "b h n d -> b h d n"), v)
            else:
                memory = torch.matmul(rearrange(k, "b h n d -> b h d n"), v)

            if norm_term is not None:
                norm_term = norm_term + k.sum(dim=-2, keepdim=True)
            else:
                norm_term = k.sum(dim=-2, keepdim=True)

            return memory, norm_term

    def forward_infini(
        self: "Attend",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        n_segments = q.shape[-2] // self.segment_len  # Assume sequence length is divisible by segment length
        q, k, v = (rearrange(t, "b h (s n) d -> b h s n d", s=n_segments) for t in (q, k, v))

        outputs = []
        memory = None
        norm_term = None
        total_segment_processed = 0
        for idx in range(n_segments):
            q_segment = q[:, :, idx, :, :]
            k_segment = k[:, :, idx, :, :]
            v_segment = v[:, :, idx, :, :]

            # Add causal mask
            attn_mask = None
            if self.causal:
                attn_mask = torch.triu(
                    torch.ones(q_segment.shape[-2], k_segment.shape[-2], device=q_segment.device),
                    diagonal=1,
                )
                attn_mask.masked_fill_(attn_mask == 1, float("-inf"))

            memory_output = self._retrieve_from_memory(q_segment, memory, norm_term)
            updated_memory, updated_norm_term = self._update_memory(
                k_segment,
                v_segment,
                memory,
                norm_term,
            )
            memory = updated_memory.detach()
            norm_term = updated_norm_term.detach()

            attn = self.forward_sdpa(q_segment, k_segment, v_segment, attn_mask=attn_mask)
            combined_output = (F.sigmoid(self.gate) * memory_output) + (1 - F.sigmoid(self.gate)) * attn
            outputs.append(combined_output)

            total_segment_processed += q_segment.shape[-2]

        out = torch.cat(outputs, dim=-2)
        return out

    def forward(
        self: "Attend",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_rotary_emb:
            q, k = self.rotary_emb(q, k)

        if self.infini:
            return self.forward_infini(q, k, v)

        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2], device=q.device), diagonal=1)
            causal_mask.masked_fill_(causal_mask == 1, float("-inf"))
        return self.forward_sdpa(q, k, v, attn_mask=causal_mask)
