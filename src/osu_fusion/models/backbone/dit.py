from typing import Dict, List

import torch
import torch.nn as nn
from einops import rearrange, repeat

from osu_fusion.modules.attention import Attention
from osu_fusion.modules.positional_embeddings import SinusoidalPositionEmbedding
from osu_fusion.modules.transformer import FeedForward
from osu_fusion.modules.utils import prob_mask_like


@torch.jit.script
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    def __init__(self: "FinalLayer", dim_h: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_h, dim_h * 2, bias=True),
        )
        self.linear = nn.Linear(dim_h, dim_h)

    def forward(self: "FinalLayer", x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiTBlock(nn.Module):
    def __init__(
        self: "DiTBlock",
        dim_h: int,
        dim_h_mult: int = 4,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 8192,
    ) -> None:
        super().__init__()

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_h, dim_h * 6, bias=True),
        )
        self.norm1 = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim_h,
            dim_head=attn_dim_head,
            heads=attn_heads,
            kv_heads=attn_kv_heads,
            context_len=attn_context_len,
        )
        self.norm2 = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim_h, dim_h_mult)

        self.gradient_checkpointing = False

    def forward_body(self: "DiTBlock", x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_ff, scale_ff, gate_ff = self.modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_ff.unsqueeze(1) * self.ff(modulate(self.norm2(x), shift_ff, scale_ff))
        return x

    def forward(self: "DiTBlock", x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, c, use_reentrant=False)
        else:
            return self.forward_body(x, c)


class DiT(nn.Module):
    def __init__(
        self: "DiT",
        dim_in_x: int,
        dim_in_a: int,
        dim_in_c: int,
        dim_h: int,
        dim_h_mult: int = 4,
        depth: int = 12,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 1,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        self.dim_in_x = dim_in_x

        self.preprocess = nn.Conv1d(dim_in_x + dim_in_a, dim_h, 7, paddding=3)
        self.postprocess = nn.Conv1d(dim_h, dim_in_x, 7, paddding=3)

        self.mlp_time = nn.Sequential(
            SinusoidalPositionEmbedding(dim_h),
            nn.Linear(dim_h, dim_h, bias=False),
            nn.SiLU(),
            nn.Linear(dim_h, dim_h, bias=False),
        )
        self.mlp_cond = nn.Sequential(
            nn.Linear(dim_in_c, dim_h),  # TODO: Better conditional embedding
            nn.SiLU(),
            nn.Linear(dim_h, dim_h),
        )
        self.null_cond = nn.Parameter(torch.randn(dim_h))
        self.feature_extractor_a = nn.Linear(dim_in_a * 2, dim_h)
        self.mlp_audio = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            nn.SiLU(),
            nn.Linear(dim_h, dim_h),
        )
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
        self.final = FinalLayer(dim_h)

        self.initialize_weights()

    def initialize_weights(self: "DiT") -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Initialize embedders
        nn.init.normal_(self.mlp_time[1].weight, std=0.02)
        nn.init.normal_(self.mlp_time[3].weight, std=0.02)
        nn.init.normal_(self.mlp_cond[0].weight, std=0.02)
        nn.init.normal_(self.mlp_cond[2].weight, std=0.02)
        nn.init.normal_(self.mlp_audio[0].weight, std=0.02)
        nn.init.normal_(self.mlp_audio[2].weight, std=0.02)

        # Zero-out adaLN layers
        for block in self.blocks:
            nn.init.zeros_(block.modulation[1].weight)
            nn.init.zeros_(block.modulation[1].bias)

        # Zero-out final layer
        nn.init.zeros_(self.final.modulation[1].weight)
        nn.init.zeros_(self.final.modulation[1].bias)

        # Zero-out postprocess
        nn.init.zeros_(self.postprocess.weight)

    def set_gradient_checkpointing(self: "DiT", value: bool) -> None:
        for name, module in self.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
                print(f"Set gradient checkpointing to {value} for {name}")

    def forward_with_cond_scale(self: "DiT", *args: List, cond_scale: float = 1.0, **kwargs: Dict) -> torch.Tensor:
        logits = self(*args, **kwargs)

        if cond_scale == 1.0:
            return logits

        null_logits = self(*args, **kwargs, cond_drop_prob=1.0)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self: "DiT",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        n = x.shape[-1]
        x = self.preprocess(torch.cat([x, a], dim=1))
        x = rearrange(x, "b d n -> b n d")

        # Statistic audio features pooling
        mean_features = a.mean(dim=-1)
        std_features = a.std(dim=-1)
        h_a = torch.cat([mean_features, std_features], dim=1)
        h_a = self.feature_extractor_a(h_a)

        cond_mask = prob_mask_like((x.shape[0],), 1.0 - cond_drop_prob, device=x.device)
        cond_mask = rearrange(cond_mask, "b -> b 1")
        null_conds = repeat(self.null_cond, "d -> b d", b=x.shape[0])
        c = self.mlp_cond(c)
        c = torch.where(cond_mask, c, null_conds)
        c = c + self.mlp_time(t) + self.mlp_audio(h_a)

        for block in self.blocks:
            x = block(x, c)

        x = self.final(x, c)
        x = rearrange(x, "b n d -> b d n")
        return self.postprocess(x[:, :, :n])
