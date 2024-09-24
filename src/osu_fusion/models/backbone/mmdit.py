from typing import Dict, List

import torch
import torch.nn as nn
from einops import rearrange, repeat

from osu_fusion.modules.attention import JointAttention
from osu_fusion.modules.positional_embeddings import SinusoidalPositionEmbedding
from osu_fusion.modules.transformer import FeedForward
from osu_fusion.modules.utils import prob_mask_like


@torch.jit.script
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MMDiTBlock(nn.Module):
    def __init__(
        self: "MMDiTBlock",
        dim_h: int,
        dim_h_mult: int = 4,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 2,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()
        # Modulation
        self.modulation_x = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_h, dim_h * 6, bias=True),
        )
        self.modulation_a = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_h, dim_h * 6, bias=True),
        )

        # OsuData branch
        self.norm1_x = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.norm2_x = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.mlp_x = FeedForward(dim_h, dim_mult=dim_h_mult)

        # Audio branch
        self.norm1_a = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.norm2_a = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.mlp_a = FeedForward(dim_h, dim_mult=dim_h_mult)

        self.attn = JointAttention(
            dim_h,
            attn_dim_head,
            attn_heads,
            attn_kv_heads,
            context_len=attn_context_len,
        )

        self.gradient_checkpointing = False

    def forward_body(
        self: "MMDiTBlock",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        # Modulation
        (
            shift_attn_x,
            scale_attn_x,
            gate_attn_x,
            shift_mlp_x,
            scale_mlp_x,
            gate_mlp_x,
        ) = self.modulation_x(c).chunk(6, dim=1)
        (
            shift_attn_a,
            scale_attn_a,
            gate_attn_a,
            shift_mlp_a,
            scale_mlp_a,
            gate_mlp_a,
        ) = self.modulation_a(c).chunk(6, dim=1)

        # Attention
        h_x = modulate(self.norm1_x(x), shift_attn_x, scale_attn_x)
        h_a = modulate(self.norm1_a(a), shift_attn_a, scale_attn_a)
        attn_out_x, attn_out_a = self.attn(h_x, h_a)

        x = x + gate_attn_x.unsqueeze(1) * attn_out_x
        a = a + gate_attn_a.unsqueeze(1) * attn_out_a

        # MLP
        x = x + gate_mlp_x.unsqueeze(1) * self.mlp_x(modulate(self.norm2_x(x), shift_mlp_x, scale_mlp_x))
        a = a + gate_mlp_a.unsqueeze(1) * self.mlp_a(modulate(self.norm2_a(a), shift_mlp_a, scale_mlp_a))

        return x, a

    def forward(
        self: "MMDiTBlock",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, a, c, use_reentrant=True)
        else:
            return self.forward_body(x, a, c)


class FinalLayer(nn.Module):
    def __init__(self: "FinalLayer", dim_h: int, dim_out: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim_h, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_h, dim_h * 2, bias=True),
        )
        self.linear = nn.Linear(dim_h, dim_out)

    def forward(self: "FinalLayer", x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class MMDiT(nn.Module):
    def __init__(
        self: "MMDiT",
        dim_in_x: int,
        dim_in_a: int,
        dim_in_c: int,
        dim_h: int,
        dim_h_mult: int = 4,
        depth: int = 12,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        attn_kv_heads: int = 2,
        attn_context_len: int = 4096,
    ) -> None:
        super().__init__()

        self.dim_h = dim_h
        self.dim_in_x = dim_in_x
        self.attn_context_len = attn_context_len * 2  # We have two modalities

        self.init_x = nn.Conv1d(dim_in_x, dim_h, 7, padding=3)
        self.init_a = nn.Conv1d(dim_in_a, dim_h, 7, padding=3)

        self.feature_extractor_a = nn.Linear(dim_in_a * 2, dim_h)
        self.mlp_a = FeedForward(dim_h, dim_mult=dim_h_mult)
        self.mlp_time = nn.Sequential(
            SinusoidalPositionEmbedding(dim_h),
            FeedForward(dim_h, dim_mult=dim_h_mult),
        )
        self.mlp_cond = nn.Sequential(
            nn.Linear(dim_in_c, dim_h),
            FeedForward(dim_h, dim_mult=dim_h_mult),
        )
        self.null_cond = nn.Parameter(torch.randn(dim_h))
        self.blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim_h,
                    dim_h_mult=dim_h_mult,
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_kv_heads=attn_kv_heads,
                    attn_context_len=self.attn_context_len,
                )
                for _ in range(depth)
            ],
        )

        self.final_layer = FinalLayer(dim_h, dim_h)
        self.out = nn.Conv1d(dim_h, dim_in_x, 1)

        self.initialize_weights()

    def initialize_weights(self: "MMDiT") -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Basic initialization for layers
        self.apply(_basic_init)

        # Initialize embedder
        nn.init.normal_(self.mlp_a[0].weight, std=0.02)
        nn.init.normal_(self.mlp_a[2].weight, std=0.02)
        nn.init.normal_(self.mlp_time[1][0].weight, std=0.02)
        nn.init.normal_(self.mlp_time[1][2].weight, std=0.02)
        nn.init.normal_(self.mlp_cond[1][0].weight, std=0.02)
        nn.init.normal_(self.mlp_cond[1][2].weight, std=0.02)

        # Zero-out adaLN layers
        for block in self.blocks:
            nn.init.zeros_(block.modulation_x[1].weight)
            nn.init.zeros_(block.modulation_x[1].bias)
            nn.init.zeros_(block.modulation_a[1].weight)
            nn.init.zeros_(block.modulation_a[1].bias)

        # Zero-out output layer
        nn.init.zeros_(self.final_layer.modulation[1].weight)
        nn.init.zeros_(self.final_layer.modulation[1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def set_gradient_checkpointing(self: "MMDiT", value: bool) -> None:
        for name, module in self.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
                print(f"Set gradient checkpointing to {value} for {name}")

    def forward_with_cond_scale(self: "MMDiT", *args: List, cond_scale: float = 1.0, **kwargs: Dict) -> torch.Tensor:
        logits = self(*args, **kwargs)

        if cond_scale == 1.0:
            return logits

        null_logits = self(*args, **kwargs, cond_drop_prob=1.0)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self: "MMDiT",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        # Statistic audio features pooling
        mean_features = a.mean(dim=-1)
        std_features = a.std(dim=-1)
        h_a = torch.cat([mean_features, std_features], dim=1)
        h_a = self.feature_extractor_a(h_a)

        x = self.init_x(x)
        a = self.init_a(a)
        x, a = (rearrange(t, "b d n -> b n d") for t in (x, a))

        # Add positional embedding and condition
        cond_mask = prob_mask_like((x.shape[0],), 1.0 - cond_drop_prob, device=x.device)
        cond_mask = rearrange(cond_mask, "b -> b 1")
        null_conds = repeat(self.null_cond, "d -> b d", b=x.shape[0])
        c = self.mlp_cond(c)
        c = torch.where(cond_mask, c, null_conds)

        c = c + self.mlp_time(t) + self.mlp_a(h_a)

        # Run the blocks
        for block in self.blocks:
            x, a = block(x, a, c)

        # Run the final layer
        x = self.final_layer(x, c)

        # Unpatchify the output
        x = rearrange(x, "b n d -> b d n")
        return self.out(x)
