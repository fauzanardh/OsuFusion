import os
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from torch.profiler import record_function

from osu_fusion.modules.attention import Attention, JointAttention, RotaryPositionEmbedding
from osu_fusion.modules.positional_embeddings import SinusoidalPositionEmbedding
from osu_fusion.modules.triton_kernels import fused_gated_residual, fused_modulate
from osu_fusion.modules.utils import prob_mask_like

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"


class FeedForward(nn.Module):
    def __init__(self: "FeedForward", dim: int, dim_mult: int = 6) -> None:
        super().__init__()
        inner_dim = int(dim * dim_mult * 2 / 3)
        inner_dim = (inner_dim + 7) // 8 * 8
        self.w1 = nn.Linear(dim, inner_dim, bias=False)
        self.w2 = nn.Linear(dim, inner_dim, bias=False)
        self.w3 = nn.Linear(inner_dim, dim, bias=False)

    def forward_body(self: "FeedForward", x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

    def forward(self: "FeedForward", x: torch.Tensor) -> torch.Tensor:
        if DEBUG:
            with record_function("FeedForward"):
                return self.forward_body(x)
        return self.forward_body(x)


class AudioPatchEmbedding(nn.Sequential):
    def __init__(self: "AudioPatchEmbedding", dim: int, dim_h: int, patch_size: int) -> None:
        super().__init__(
            Rearrange("b (n p) d -> b n (p d)", p=patch_size),
            nn.Linear(dim * patch_size, dim_h),
            nn.LayerNorm(dim_h),
        )


class BeatmapPatchEmbedding(nn.Sequential):
    def __init__(self: "BeatmapPatchEmbedding", dim: int, dim_h: int, patch_size: int) -> None:
        super().__init__(
            Rearrange("b (n p) d -> b n (p d)", p=patch_size),
            nn.Linear(dim * patch_size, dim_h),
            nn.LayerNorm(dim_h),
        )


class FinalUnpatchLayer(nn.Module):
    def __init__(self: "FinalUnpatchLayer", dim_h: int, dim_out: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim_h, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_h, dim_h * 2, bias=True),
        )
        self.out = nn.Linear(dim_h, dim_out * patch_size)
        self.unpatch = Rearrange("b n (p d) -> b (n p) d", p=patch_size, d=dim_out)

    def forward_body(self: "FinalUnpatchLayer", x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        x = fused_modulate(self.norm(x), shift, scale)
        x = self.out(x)
        return self.unpatch(x)

    def forward(self: "FinalUnpatchLayer", x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if DEBUG:
            with record_function("FinalUnpatchLayer"):
                return self.forward_body(x, c)
        return self.forward_body(x, c)


class MMDiTBlock(nn.Module):
    def __init__(
        self: "MMDiTBlock",
        dim_h: int,
        dim_h_mult: int = 6,
        attn_dim_head: int = 64,
        attn_heads: int = 6,
        attn_context_len: int = 8192,
        rotary_emb: RotaryPositionEmbedding = None,
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
            rotary_emb=rotary_emb,
            context_len=attn_context_len,
        )

        self.gradient_checkpointing = False

    def forward_body(
        self: "MMDiTBlock",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
        mask_x: Optional[torch.Tensor] = None,
        mask_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Modulation
        (
            shift_attn_x,
            scale_attn_x,
            gate_attn_x,
            shift_mlp_x,
            scale_mlp_x,
            gate_mlp_x,
        ) = self.modulation_x(c).chunk(6, dim=-1)
        (
            shift_attn_a,
            scale_attn_a,
            gate_attn_a,
            shift_mlp_a,
            scale_mlp_a,
            gate_mlp_a,
        ) = self.modulation_a(c).chunk(6, dim=-1)

        # Attention
        h_x = fused_modulate(self.norm1_x(x), shift_attn_x, scale_attn_x)
        h_a = fused_modulate(self.norm1_a(a), shift_attn_a, scale_attn_a)
        attn_out_x, attn_out_a = self.attn(h_x, h_a, mask_x=mask_x, mask_a=mask_a)

        x = fused_gated_residual(x, gate_attn_x, attn_out_x)
        a = fused_gated_residual(a, gate_attn_a, attn_out_a)

        # MLP
        x = fused_gated_residual(x, gate_mlp_x, self.mlp_x(fused_modulate(self.norm2_x(x), shift_mlp_x, scale_mlp_x)))
        a = fused_gated_residual(a, gate_mlp_a, self.mlp_a(fused_modulate(self.norm2_a(a), shift_mlp_a, scale_mlp_a)))

        return x, a

    def forward(
        self: "MMDiTBlock",
        x: torch.Tensor,
        a: torch.Tensor,
        c: torch.Tensor,
        mask_x: Optional[torch.Tensor] = None,
        mask_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self.forward_body,
                x,
                a,
                c,
                mask_x,
                mask_a,
                use_reentrant=True,
            )
        return self.forward_body(x, a, c, mask_x=mask_x, mask_a=mask_a)


class DiTBlock(nn.Module):
    def __init__(
        self: "DiTBlock",
        dim_h: int,
        dim_h_mult: int = 6,
        attn_dim_head: int = 64,
        attn_heads: int = 6,
        attn_context_len: int = 8192,
        rotary_emb: RotaryPositionEmbedding = None,
    ) -> None:
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_h, dim_h * 6, bias=True),
        )
        self.norm1 = nn.LayerNorm(dim_h, elementwise_affine=False)
        self.attn = Attention(
            dim_h,
            dim_head=attn_dim_head,
            heads=attn_heads,
            rotary_emb=rotary_emb,
            context_len=attn_context_len,
        )
        self.norm2 = nn.LayerNorm(dim_h, elementwise_affine=False)
        self.ff = FeedForward(dim_h, dim_h_mult)

        self.gradient_checkpointing = False

    def forward_body(
        self: "DiTBlock",
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_ff,
            scale_ff,
            gate_ff,
        ) = self.modulation(c).chunk(6, dim=-1)

        x = fused_gated_residual(
            x,
            gate_msa,
            self.attn(fused_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask),
        )
        x = fused_gated_residual(x, gate_ff, self.ff(fused_modulate(self.norm2(x), shift_ff, scale_ff)))
        return x

    def forward(
        self: "DiTBlock",
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self.forward_body,
                x,
                c,
                attn_mask,
                use_reentrant=True,
            )
        return self.forward_body(x, c, attn_mask=attn_mask)


class DiT(nn.Module):
    def __init__(
        self: "DiT",
        dim_in_x: int,
        dim_in_a: int,
        dim_in_c: int,
        dim_h: int = 384,
        dim_h_mult: int = 6,
        dim_t: int = 256,
        beatmap_patch_size: int = 4,
        audio_patch_size: int = 4,
        mmdit_depth: int = 9,
        dit_depth: int = 3,
        attn_dim_head: int = 64,
        attn_heads: int = 6,
        attn_context_len: int = 8192,
        # num_descriptors: int = 0,
        # num_mappers: int = 0,
    ) -> None:
        super().__init__()
        self.attn_heads = attn_heads
        self.audio_patch_size = audio_patch_size
        self.beatmap_patch_size = beatmap_patch_size
        # self.num_descriptors = num_descriptors
        # self.num_mappers = num_mappers

        self.x_embed = BeatmapPatchEmbedding(dim_in_x, dim_h, beatmap_patch_size)
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

        # if num_descriptors > 0:
        #     self.descriptor_proj = nn.Sequential(
        #         nn.Linear(num_descriptors, dim_h),
        #         nn.SiLU(),
        #         nn.Linear(dim_h, dim_h),
        #     )
        #     self.null_descriptor = nn.Parameter(torch.randn(dim_h))
        # else:
        #     self.descriptor_proj = None
        #     self.null_descriptor = None

        # if num_mappers > 0:
        #     self.mapper_proj = nn.Sequential(
        #         nn.Linear(num_mappers + 1, dim_h),  # +1 for unknown
        #         nn.SiLU(),
        #         nn.Linear(dim_h, dim_h),
        #     )
        #     self.null_mapper = nn.Parameter(torch.randn(dim_h))
        # else:
        #     self.mapper_proj = None
        #     self.null_mapper = None

        self.shared_rotary_emb = RotaryPositionEmbedding(attn_dim_head, scale_base=attn_context_len)
        self.mmdit_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim_h,
                    dim_h_mult=dim_h_mult,
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_context_len=attn_context_len,
                    rotary_emb=self.shared_rotary_emb,
                )
                for _ in range(mmdit_depth)
            ],
        )
        self.dit_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim_h,
                    dim_h_mult=dim_h_mult,
                    attn_dim_head=attn_dim_head,
                    attn_heads=attn_heads,
                    attn_context_len=attn_context_len,
                    rotary_emb=self.shared_rotary_emb,
                )
                for _ in range(dit_depth)
            ],
        )

        self.final = FinalUnpatchLayer(dim_h, dim_in_x, beatmap_patch_size)

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

        # if self.descriptor_proj is not None:
        #     nn.init.normal_(self.descriptor_proj[0].weight, std=0.02)
        #     nn.init.normal_(self.descriptor_proj[2].weight, std=0.02)
        # if self.mapper_proj is not None:
        #     nn.init.normal_(self.mapper_proj[0].weight, std=0.02)
        #     nn.init.normal_(self.mapper_proj[2].weight, std=0.02)

        for block in self.mmdit_blocks:
            nn.init.zeros_(block.modulation_x[1].weight)
            nn.init.zeros_(block.modulation_x[1].bias)
            nn.init.zeros_(block.modulation_a[1].weight)
            nn.init.zeros_(block.modulation_a[1].bias)

        for block in self.dit_blocks:
            nn.init.zeros_(block.modulation[1].weight)
            nn.init.zeros_(block.modulation[1].bias)

        nn.init.zeros_(self.final.modulation[1].weight)
        nn.init.zeros_(self.final.modulation[1].bias)

        nn.init.zeros_(self.final.out.weight)
        nn.init.zeros_(self.final.out.bias)

    def compile_blocks(self: "DiT", dynamic: bool = True) -> None:
        for block in self.mmdit_blocks:
            block.forward = torch.compile(block.forward, dynamic=dynamic)
            print(f"Compiled MMDiTBlock with dynamic={dynamic}")
        for block in self.dit_blocks:
            block.forward = torch.compile(block.forward, dynamic=dynamic)
            print(f"Compiled DiTBlock with dynamic={dynamic}")

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
        # descriptors: Optional[torch.Tensor] = None,
        # mappers: Optional[torch.Tensor] = None,
        cond_scale: float = 1.0,
    ) -> torch.Tensor:
        logits = self.forward(x, a, t, c, cond_drop_prob=0.0)
        if cond_scale == 1.0:
            return logits
        null_logits = self.forward(x, a, t, c, cond_drop_prob=1.0)
        return null_logits + (logits - null_logits) * cond_scale

    def _build_patch_masks(
        self: "DiT",
        orig_lens: torch.Tensor,
        n_x_patches: int,
        n_a_patches: int,
        device: torch.device,
    ) -> tuple:
        # (B, n_x_patches)
        patch_starts_x = torch.arange(n_x_patches, device=device) * self.beatmap_patch_size
        mask_x = patch_starts_x.unsqueeze(0) < orig_lens.unsqueeze(1).to(device)

        # (B, n_a_patches)
        patch_starts_a = torch.arange(n_a_patches, device=device) * self.audio_patch_size
        mask_a = patch_starts_a.unsqueeze(0) < orig_lens.unsqueeze(1).to(device)

        return mask_x, mask_a

    def _mask_to_attn_bias(
        self: "DiT",
        mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        attn_bias = torch.zeros(mask.shape, device=mask.device, dtype=dtype)
        attn_bias = attn_bias.masked_fill(~mask, float("-inf"))
        return attn_bias[:, None, None, :]

    def forward(
        self: "DiT",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        # descriptors: Optional[torch.Tensor] = None,
        # mappers: Optional[torch.Tensor] = None,
        cond_drop_prob: float = 0.0,
        orig_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_x_len = x.shape[1]

        # Pad beatmap
        x_pad = (self.beatmap_patch_size - (orig_x_len % self.beatmap_patch_size)) % self.beatmap_patch_size
        x = F.pad(x, (0, 0, 0, x_pad))
        x = self.x_embed(x)

        # Pad audio
        orig_a_len = a.shape[1]
        a_pad = (self.audio_patch_size - (orig_a_len % self.audio_patch_size)) % self.audio_patch_size
        a = F.pad(a, (0, 0, 0, a_pad))
        a = self.a_patch(a)

        # Construct masks
        mask_x, mask_a, attn_mask_x = None, None, None
        if orig_lens is not None:
            mask_x, mask_a = self._build_patch_masks(
                orig_lens,
                x.shape[1],
                a.shape[1],
                x.device,
            )
            attn_mask_x = self._mask_to_attn_bias(mask_x, x.dtype)

        # Global conditioning
        b = c.shape[0]
        cond_mask = prob_mask_like((b,), 1.0 - cond_drop_prob, device=c.device)
        cond_mask = rearrange(cond_mask, "b -> b 1")

        # Base conditioning (CS, AR, OD, HP, SR, SV, STR, era)
        null_conds = repeat(self.null_cond, "d -> b d", b=b)
        c_global = self.cond_mlp(c)
        c_global = torch.where(cond_mask, c_global, null_conds)

        # if self.descriptor_proj is not None:
        #     null_desc = repeat(self.null_descriptor, "d -> b d", b=b)
        #     if descriptors is not None:
        #         desc_mask = prob_mask_like((b,), 1.0 - cond_drop_prob, device=c.device)
        #         desc_mask = rearrange(desc_mask, "b -> b 1")
        #         desc_emb = self.descriptor_proj(descriptors)
        #         c_global = c_global + torch.where(desc_mask, desc_emb, null_desc)
        #     else:
        #         c_global = c_global + null_desc

        # if self.mapper_proj is not None:
        #     null_map = repeat(self.null_mapper, "d -> b d", b=b)
        #     if mappers is not None:
        #         mapper_mask = prob_mask_like((b,), 1.0 - cond_drop_prob, device=c.device)
        #         mapper_mask = rearrange(mapper_mask, "b -> b 1")
        #         map_emb = self.mapper_proj(mappers)
        #         c_global = c_global + torch.where(mapper_mask, map_emb, null_map)
        #     else:
        #         c_global = c_global + null_map

        # Add timestep
        c_global = c_global + self.time_mlp(t)
        c_global = c_global.unsqueeze(1)

        for mmdit_block in self.mmdit_blocks:
            x, a = mmdit_block(x, a, c_global, mask_x=mask_x, mask_a=mask_a)

        for dit_block in self.dit_blocks:
            x = dit_block(x, c_global, attn_mask=attn_mask_x)

        x = self.final(x, c_global)
        return x[:, :orig_x_len, :]
