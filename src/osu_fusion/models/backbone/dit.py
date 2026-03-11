import os

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from torch.profiler import record_function

from osu_fusion.modules.attention import Attention, CrossAttention
from osu_fusion.modules.positional_embeddings import SinusoidalPositionEmbedding
from osu_fusion.modules.utils import dummy_context_manager, prob_mask_like


DEBUG = os.environ.get("DEBUG", False)


@torch.jit.script
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class FeedForward(nn.Sequential):
    def __init__(self: "FeedForward", dim: int, dim_mult: int = 2) -> None:
        inner_dim = dim * dim_mult
        super().__init__(
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, dim),
        )

    def forward(self: "FeedForward", x: torch.Tensor) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("FeedForward")
        with context_manager:
            return super().forward(x)


class AudioPatchEmbedding(nn.Sequential):
    def __init__(self: "AudioPatchEmbedding", dim: int, dim_h: int, patch_size: int) -> None:
        super().__init__(
            Rearrange("b (n p) d -> b n (p d)", p=patch_size),
            nn.Linear(dim * patch_size, dim_h),
            nn.LayerNorm(dim_h),
        )


class BeatmapEmbedding(nn.Sequential):
    def __init__(self: "BeatmapEmbedding", dim: int, dim_h: int) -> None:
        super().__init__(
            nn.Linear(dim, dim_h),
            nn.LayerNorm(dim_h),
        )


class DiTBlock(nn.Module):
    def __init__(
        self: "DiTBlock",
        dim_h: int,
        dim_h_mult: int = 4,
        attn_dim_head: int = 64,
        attn_heads: int = 16,
        attn_kv_heads: int = 8,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_h, dim_h * 9, bias=True),
        )
        self.norm1 = nn.LayerNorm(dim_h, elementwise_affine=False)
        self.attn = Attention(
            dim_h,
            dim_head=attn_dim_head,
            heads=attn_heads,
            kv_heads=attn_kv_heads,
            context_len=attn_context_len,
        )
        self.norm2 = nn.LayerNorm(dim_h, elementwise_affine=False)
        self.cross_attn = CrossAttention(
            dim_h,
            dim_head=attn_dim_head,
            heads=attn_heads,
            kv_heads=attn_kv_heads,
        )
        self.norm3 = nn.LayerNorm(dim_h, elementwise_affine=False)
        self.ff = FeedForward(dim_h, dim_h_mult)

        self.gradient_checkpointing = False

    def forward_body(self: "DiTBlock", x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_ff,
            scale_ff,
            gate_ff,
        ) = self.modulation(c).chunk(9, dim=-1)

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_cross * self.cross_attn(modulate(self.norm2(x), shift_cross, scale_cross), a)
        x = x + gate_ff * self.ff(modulate(self.norm3(x), shift_ff, scale_ff))
        return x

    def forward(self: "DiTBlock", x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, a, c, use_reentrant=True)
        else:
            return self.forward_body(x, a, c)


class FinalLayer(nn.Module):
    def __init__(self: "FinalLayer", dim_h: int, dim_out: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim_h, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_h, dim_h * 2, bias=True),
        )
        self.out = nn.Linear(dim_h, dim_out)

    def forward(self: "FinalLayer", x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.out(x)


class LengthPredictor(nn.Module):
    def __init__(self: "LengthPredictor", dim_a: int, dim_c: int, dim_h: int = 256) -> None:
        super().__init__()
        self.audio_pool = nn.Sequential(
            nn.Linear(dim_a, dim_h),
            nn.SiLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim_h + dim_c, dim_h),
            nn.SiLU(),
            nn.Linear(dim_h, 1),
        )

    def forward(self: "LengthPredictor", a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # a: (B, T, dim_a) — raw audio spectrogram
        # c: (B, dim_c) — context vector
        a_pooled = self.audio_pool(a.mean(dim=1))  # (B, dim_h)
        return self.mlp(torch.cat([a_pooled, c], dim=-1)).squeeze(-1)  # (B,)


class DiT(nn.Module):
    def __init__(
        self: "DiT",
        dim_in_x: int,
        dim_in_a: int,
        dim_in_c: int,
        dim_h: int,
        dim_h_mult: int = 4,
        dim_t: int = 256,
        audio_patch_size: int = 8,
        depth: int = 24,
        attn_dim_head: int = 64,
        attn_heads: int = 16,
        attn_kv_heads: int = 8,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.audio_patch_size = audio_patch_size

        self.x_embed = BeatmapEmbedding(dim_in_x, dim_h)
        self.a_patch = AudioPatchEmbedding(dim_in_a, dim_h, audio_patch_size)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(dim_t),
            nn.Linear(dim_t, dim_h),
            nn.SiLU(),
            nn.Linear(dim_h, dim_h),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(dim_in_c, dim_h),
            nn.SiLU(),
            nn.Linear(dim_h, dim_h),
        )
        self.null_cond = nn.Parameter(torch.randn(dim_h))

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim_h,
                    dim_h_mult=dim_h_mult,
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_kv_heads=attn_kv_heads,
                    attn_context_len=attn_context_len,
                )
                for _ in range(depth)
            ],
        )
        self.final = FinalLayer(dim_h, dim_in_x)

        self.initialize_weights()

    def initialize_weights(self: "DiT") -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        nn.init.normal_(self.time_mlp[1].weight, std=0.02)
        nn.init.normal_(self.time_mlp[3].weight, std=0.02)
        nn.init.normal_(self.cond_mlp[0].weight, std=0.02)
        nn.init.normal_(self.cond_mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.modulation[1].weight)
            nn.init.zeros_(block.modulation[1].bias)

        nn.init.zeros_(self.final.modulation[1].weight)
        nn.init.zeros_(self.final.modulation[1].bias)

    def set_gradient_checkpointing(self: "DiT", value: bool) -> None:
        for name, module in self.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
                print(f"Set gradient checkpointing to {value} for {name}")

    def forward_with_cond_scale(
        self: "DiT",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        cond_scale: float = 1.0,
    ) -> torch.Tensor:
        logits = self.forward(x, a, t, c, cond_drop_prob=0.0)

        if cond_scale == 1.0:
            return logits

        null_logits = self.forward(x, a, t, c, cond_drop_prob=1.0)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self: "DiT",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        # Beatmap: embed directly (no patching, each event = 1 token)
        # x: (B, N, dim_in_x)
        x = self.x_embed(x)

        # Audio: pad for patch alignment and embed
        # a: (B, T, dim_in_a)
        a_t = a.shape[1]
        a_pad = (self.audio_patch_size - (a_t % self.audio_patch_size)) % self.audio_patch_size
        a = F.pad(a, (0, 0, 0, a_pad))
        a_tokens = self.a_patch(a)

        # Global conditioning: timestep + context with CFG
        cond_mask = prob_mask_like((c.shape[0],), 1.0 - cond_drop_prob, device=c.device)
        cond_mask = rearrange(cond_mask, "b -> b 1")
        null_conds = repeat(self.null_cond, "d -> b d", b=c.shape[0])
        c_global = self.cond_mlp(c)
        c_global = torch.where(cond_mask, c_global, null_conds)
        c_global = c_global + self.time_mlp(t)
        c_global = c_global.unsqueeze(1)  # (B, 1, dim_h) for adaLN broadcast

        # Main DiT blocks: self-attention on beatmap + cross-attention into audio
        for block in self.blocks:
            x = block(x, a_tokens, c_global)

        # Project back to beatmap channels
        x = self.final(x, c_global)
        return x
