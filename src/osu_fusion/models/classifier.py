from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn import functional as F

from osu_fusion.data.const import AUDIO_DIM, CONTEXT_DIM
from osu_fusion.data.descriptors import NUM_DESCRIPTORS
from osu_fusion.data.encode import SEQ_DIM
from osu_fusion.models.backbone.dit import FeedForward
from osu_fusion.modules.attention import Attention, JointAttention, RotaryPositionEmbedding


class DropPath(nn.Module):
    def __init__(self: "DropPath", drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self: "DropPath", x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep_prob)
        return x * mask / keep_prob


class JointClassifierBlock(nn.Module):
    def __init__(
        self: "JointClassifierBlock",
        dim_h: int,
        dim_h_mult: int = 4,
        attn_dim_head: int = 64,
        attn_heads: int = 4,
        attn_context_len: int = 4096,
        rotary_emb: Optional[RotaryPositionEmbedding] = None,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1_x = nn.LayerNorm(dim_h)
        self.norm2_x = nn.LayerNorm(dim_h)
        self.mlp_x = FeedForward(dim_h, dim_mult=dim_h_mult)

        self.norm1_a = nn.LayerNorm(dim_h)
        self.norm2_a = nn.LayerNorm(dim_h)
        self.mlp_a = FeedForward(dim_h, dim_mult=dim_h_mult)

        self.attn = JointAttention(
            dim_h,
            attn_dim_head,
            attn_heads,
            rotary_emb=rotary_emb,
            context_len=attn_context_len,
        )

        self.drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

        self.gradient_checkpointing = False

    def forward_body(
        self: "JointClassifierBlock",
        x: torch.Tensor,
        a: torch.Tensor,
        mask_x: Optional[torch.Tensor] = None,
        mask_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out_x, attn_out_a = self.attn(self.norm1_x(x), self.norm1_a(a), mask_x=mask_x, mask_a=mask_a)
        x = x + self.drop_path(self.drop(attn_out_x))
        a = a + self.drop_path(self.drop(attn_out_a))

        x = x + self.drop_path(self.drop(self.mlp_x(self.norm2_x(x))))
        a = a + self.drop_path(self.drop(self.mlp_a(self.norm2_a(a))))

        return x, a

    def forward(
        self: "JointClassifierBlock",
        x: torch.Tensor,
        a: torch.Tensor,
        mask_x: Optional[torch.Tensor] = None,
        mask_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self.forward_body,
                x,
                a,
                mask_x,
                mask_a,
                use_reentrant=True,
            )
        return self.forward_body(x, a, mask_x=mask_x, mask_a=mask_a)


class ClassifierBlock(nn.Module):
    def __init__(
        self: "ClassifierBlock",
        dim_h: int,
        dim_h_mult: int = 4,
        attn_dim_head: int = 64,
        attn_heads: int = 4,
        attn_context_len: int = 4096,
        rotary_emb: Optional[RotaryPositionEmbedding] = None,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_h)
        self.attn = Attention(
            dim_h,
            dim_head=attn_dim_head,
            heads=attn_heads,
            rotary_emb=rotary_emb,
            context_len=attn_context_len,
        )
        self.norm2 = nn.LayerNorm(dim_h)
        self.ff = FeedForward(dim_h, dim_mult=dim_h_mult)

        self.drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

        self.gradient_checkpointing = False

    def forward_body(
        self: "ClassifierBlock",
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.drop(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path(self.drop(self.ff(self.norm2(x))))
        return x

    def forward(
        self: "ClassifierBlock",
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self.forward_body,
                x,
                attn_mask,
                use_reentrant=True,
            )
        return self.forward_body(x, attn_mask=attn_mask)


class PatchEmbedding(nn.Sequential):
    def __init__(self: "PatchEmbedding", dim_in: int, dim_h: int, patch_size: int) -> None:
        super().__init__(
            Rearrange("b (n p) d -> b n (p d)", p=patch_size),
            nn.Linear(dim_in * patch_size, dim_h),
            nn.LayerNorm(dim_h),
        )


class MultiScaleAttentivePooling(nn.Module):
    def __init__(self: "MultiScaleAttentivePooling", dim_h: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim_h, dim_h),
                    nn.Tanh(),
                    nn.Linear(dim_h, 1, bias=False),
                )
                for _ in range(num_heads)
            ],
        )
        self.proj = nn.Sequential(
            nn.Linear(dim_h * num_heads, dim_h),
            nn.Dropout(dropout),
        )

    def forward(
        self: "MultiScaleAttentivePooling",
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, N, D), mask: (B, N) bool
        pooled = []
        for head in self.heads:
            scores = head(x).squeeze(-1)  # (B, N)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, N, 1)
            pooled.append((weights * x).sum(dim=1))  # (B, D)
        return self.proj(torch.cat(pooled, dim=-1))  # (B, D)


@dataclass
class ClassifierConfig:
    dim_h: int = 192
    dim_h_mult: int = 4
    patch_size: int = 16
    joint_depth: int = 2
    self_depth: int = 1
    attn_dim_head: int = 64
    attn_heads: int = 4
    attn_context_len: int = 4096
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    pool_heads: int = 4


ClassifierConfig_S = ClassifierConfig()
ClassifierConfig_M = ClassifierConfig(dim_h=256, joint_depth=4, self_depth=2, attn_heads=4)
ClassifierConfig_L = ClassifierConfig(dim_h=384, joint_depth=8, self_depth=4, attn_heads=6)


class OsuFusionClassifier(nn.Module):
    def __init__(
        self: "OsuFusionClassifier",
        dim_h: int = 192,
        dim_h_mult: int = 4,
        patch_size: int = 16,
        joint_depth: int = 2,
        self_depth: int = 1,
        attn_dim_head: int = 64,
        attn_heads: int = 4,
        attn_context_len: int = 4096,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        pool_heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.x_embed = PatchEmbedding(SEQ_DIM, dim_h, patch_size)
        self.a_embed = PatchEmbedding(AUDIO_DIM, dim_h, patch_size)

        self.cond_mlp = nn.Sequential(
            nn.Linear(CONTEXT_DIM, dim_h),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_h, dim_h),
        )

        self.rotary_emb = RotaryPositionEmbedding(attn_dim_head, scale_base=attn_context_len)

        # Linearly increasing drop path rates across all blocks
        total_depth = joint_depth + self_depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.joint_blocks = nn.ModuleList(
            [
                JointClassifierBlock(
                    dim_h,
                    dim_h_mult=dim_h_mult,
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_context_len=attn_context_len,
                    rotary_emb=self.rotary_emb,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(joint_depth)
            ],
        )

        self.self_blocks = nn.ModuleList(
            [
                ClassifierBlock(
                    dim_h,
                    dim_h_mult=dim_h_mult,
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_context_len=attn_context_len,
                    rotary_emb=self.rotary_emb,
                    dropout=dropout,
                    drop_path=dpr[joint_depth + i],
                )
                for i in range(self_depth)
            ],
        )

        self.pool = MultiScaleAttentivePooling(dim_h, num_heads=pool_heads, dropout=dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(dim_h),
            nn.Dropout(dropout),
            nn.Linear(dim_h, NUM_DESCRIPTORS),
        )

        self.initialize_weights()

    def initialize_weights(self: "OsuFusionClassifier") -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        nn.init.normal_(self.cond_mlp[0].weight, std=0.02)
        nn.init.normal_(self.cond_mlp[3].weight, std=0.02)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def set_gradient_checkpointing(self: "OsuFusionClassifier", value: bool) -> None:
        for name, module in self.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
                print(f"Set gradient checkpointing to {value} for {name}")

    def _build_patch_mask(
        self: "OsuFusionClassifier",
        orig_lens: torch.Tensor,
        n_patches: int,
        device: torch.device,
    ) -> torch.Tensor:
        patch_starts = torch.arange(n_patches, device=device) * self.patch_size
        return patch_starts.unsqueeze(0) < orig_lens.unsqueeze(1).to(device)

    def _mask_to_attn_bias(
        self: "OsuFusionClassifier",
        mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        attn_bias = torch.zeros(mask.shape, device=mask.device, dtype=dtype)
        attn_bias = attn_bias.masked_fill(~mask, float("-inf"))
        return attn_bias[:, None, None, :]

    def forward(
        self: "OsuFusionClassifier",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
        orig_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = x.shape[0]
        orig_len = x.shape[1]

        x_pad = (self.patch_size - (orig_len % self.patch_size)) % self.patch_size
        x = F.pad(x, (0, 0, 0, x_pad))
        x = self.x_embed(x)  # (B, N_x, dim_h)

        a_pad = (self.patch_size - (a.shape[1] % self.patch_size)) % self.patch_size
        a = F.pad(a, (0, 0, 0, a_pad))
        a = self.a_embed(a)  # (B, N_a, dim_h)

        n_x_patches = x.shape[1]
        n_a_patches = a.shape[1]

        mask_x, mask_a = None, None
        if orig_lens is not None:
            mask_x = self._build_patch_mask(orig_lens, n_x_patches, x.device)
            mask_a = self._build_patch_mask(orig_lens, n_a_patches, a.device)

        for block in self.joint_blocks:
            x, a = block(x, a, mask_x=mask_x, mask_a=mask_a)

        cond = self.cond_mlp(c).unsqueeze(1)  # (B, 1, dim_h)
        tokens = torch.cat([cond, x], dim=1)  # (B, 1+N_x, dim_h)

        attn_mask = None
        pool_mask = None
        if orig_lens is not None:
            prefix_mask = torch.ones(b, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([prefix_mask, mask_x], dim=1)
            attn_mask = self._mask_to_attn_bias(full_mask, tokens.dtype)
            pool_mask = full_mask

        for block in self.self_blocks:
            tokens = block(tokens, attn_mask=attn_mask)

        pooled = self.pool(tokens, mask=pool_mask)  # (B, dim_h)

        return self.head(pooled)  # (B, NUM_DESCRIPTORS)
