import os
from typing import Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from torch.profiler import record_function

from osu_fusion.data.const import AUDIO_DIM, NUM_CONTINUOUS_CONDS, NUM_ERAS
from osu_fusion.data.encode import SEQ_DIM
from osu_fusion.modules.attention import Attention, JointAttention, RotaryPositionEmbedding
from osu_fusion.modules.positional_embeddings import LearnedSinusoidalPosEmb, SinusoidalPositionEmbedding
from osu_fusion.modules.utils import prob_mask_like

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


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
        x = modulate(self.norm(x), shift, scale)
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
        h_x = modulate(self.norm1_x(x), shift_attn_x, scale_attn_x)
        h_a = modulate(self.norm1_a(a), shift_attn_a, scale_attn_a)
        attn_out_x, attn_out_a = self.attn(h_x, h_a, mask_x=mask_x, mask_a=mask_a)

        x = x + gate_attn_x * attn_out_x
        a = a + gate_attn_a * attn_out_a

        # MLP
        x = x + gate_mlp_x * self.mlp_x(modulate(self.norm2_x(x), shift_mlp_x, scale_mlp_x))
        a = a + gate_mlp_a * self.mlp_a(modulate(self.norm2_a(a), shift_mlp_a, scale_mlp_a))

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
                use_reentrant=False,
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

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_ff * self.ff(modulate(self.norm2(x), shift_ff, scale_ff))
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
                use_reentrant=False,
            )
        return self.forward_body(x, c, attn_mask=attn_mask)


class DiT(nn.Module):
    def __init__(
        self: "DiT",
        dim_h: int = 384,
        dim_h_mult: int = 6,
        dim_t: int = 256,
        dim_cond_fourier: int = 32,
        beatmap_patch_size: int = 4,
        audio_patch_size: int = 4,
        mmdit_depth: int = 9,
        dit_depth: int = 3,
        attn_dim_head: int = 64,
        attn_heads: int = 6,
        attn_context_len: int = 8192,
        num_descriptors: int = 0,
        num_mappers: int = 0,
    ) -> None:
        super().__init__()
        self.attn_heads = attn_heads
        self.audio_patch_size = audio_patch_size
        self.beatmap_patch_size = beatmap_patch_size
        self.num_descriptors = num_descriptors
        self.num_mappers = num_mappers
        self.dim_cond_fourier = dim_cond_fourier

        self.x_embed = BeatmapPatchEmbedding(SEQ_DIM, dim_h, beatmap_patch_size)
        self.a_patch = AudioPatchEmbedding(AUDIO_DIM, dim_h, audio_patch_size)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(dim_t),
            nn.Linear(dim_t, dim_h),
            nn.SiLU(),
            nn.Linear(dim_h, dim_h),
        )

        self.cond_fourier_embeds = nn.ModuleList(
            [LearnedSinusoidalPosEmb(dim_cond_fourier) for _ in range(NUM_CONTINUOUS_CONDS)],
        )
        self.era_embed = nn.Embedding(NUM_ERAS + 1, dim_cond_fourier)

        self.null_cond_embeds = nn.ParameterList(
            [nn.Parameter(torch.randn(dim_cond_fourier)) for _ in range(NUM_CONTINUOUS_CONDS)],
        )
        self.null_era_embed = nn.Parameter(torch.randn(dim_cond_fourier))

        if num_descriptors > 0:
            self.descriptor_proj = nn.Linear(num_descriptors, dim_cond_fourier)
            self.null_descriptor_embed = nn.Parameter(torch.randn(dim_cond_fourier))
        else:
            self.descriptor_proj = None
            self.null_descriptor_embed = None

        if num_mappers > 0:
            self.mapper_proj = nn.Linear(num_mappers + 1, dim_cond_fourier)  # +1 for unknown
            self.null_mapper_embed = nn.Parameter(torch.randn(dim_cond_fourier))
        else:
            self.mapper_proj = None
            self.null_mapper_embed = None

        # Joint MLP: concatenated per-condition Fourier features + era + descriptors + mappers → dim_h
        num_cond_slots = NUM_CONTINUOUS_CONDS + 1  # +1 for era
        if num_descriptors > 0:
            num_cond_slots += 1
        if num_mappers > 0:
            num_cond_slots += 1
        joint_input_dim = num_cond_slots * dim_cond_fourier
        self.cond_joint_mlp = nn.Sequential(
            nn.Linear(joint_input_dim, dim_h * 2),
            nn.SiLU(),
            nn.Linear(dim_h * 2, dim_h),
        )

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

        self.final = FinalUnpatchLayer(dim_h, SEQ_DIM, beatmap_patch_size)

        self.initialize_weights()

    def initialize_weights(self: "DiT") -> None:
        # def _basic_init(module: nn.Module) -> None:
        #     if isinstance(module, (nn.Linear, nn.Conv1d)):
        #         nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.zeros_(module.bias)

        # self.apply(_basic_init)

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
        descriptors: Optional[torch.Tensor] = None,
        mappers: Optional[torch.Tensor] = None,
        cond_scale: float = 1.0,
    ) -> torch.Tensor:
        logits = self.forward(x, a, t, c, descriptors=descriptors, mappers=mappers, cond_drop_prob=0.0)
        if cond_scale == 1.0:
            return logits
        null_logits = self.forward(x, a, t, c, descriptors=descriptors, mappers=mappers, cond_drop_prob=1.0)
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
        descriptors: Optional[torch.Tensor] = None,
        mappers: Optional[torch.Tensor] = None,
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

        # Global conditioning — sample-level all-or-nothing dropout for CFG alignment
        b = c.shape[0]
        # Single mask per sample: when True the sample keeps all conditions, when False all go null
        cond_keep_mask = prob_mask_like((b,), 1.0 - cond_drop_prob, device=c.device)  # (B,)
        keep = cond_keep_mask.unsqueeze(-1)  # (B, 1) for broadcasting

        cond_features = []
        for i, cond_fourier in enumerate(self.cond_fourier_embeds):
            feat = cond_fourier(c[:, i])  # (B, dim_cond_fourier)
            null_i = self.null_cond_embeds[i].unsqueeze(0).expand(b, -1)  # (B, dim_cond_fourier)
            feat = torch.where(keep, feat, null_i)
            cond_features.append(feat)

        era_raw = c[:, NUM_CONTINUOUS_CONDS].long()
        era_idx = torch.where(era_raw >= 0, era_raw, torch.full_like(era_raw, NUM_ERAS))
        era_feat = self.era_embed(era_idx)  # (B, dim_cond_fourier)
        null_era = self.null_era_embed.unsqueeze(0).expand(b, -1)
        era_feat = torch.where(keep, era_feat, null_era)
        cond_features.append(era_feat)

        # Descriptor conditioning (multi-hot → linear projection)
        if self.descriptor_proj is not None:
            if descriptors is not None:
                desc_feat = self.descriptor_proj(descriptors)  # (B, dim_cond_fourier)
                null_desc = self.null_descriptor_embed.unsqueeze(0).expand(b, -1)
                desc_feat = torch.where(keep, desc_feat, null_desc)
            else:
                desc_feat = self.null_descriptor_embed.unsqueeze(0).expand(b, -1)
            cond_features.append(desc_feat)

        # Mapper conditioning (multi-hot → linear projection)
        if self.mapper_proj is not None:
            if mappers is not None:
                map_feat = self.mapper_proj(mappers)  # (B, dim_cond_fourier)
                null_map = self.null_mapper_embed.unsqueeze(0).expand(b, -1)
                map_feat = torch.where(keep, map_feat, null_map)
            else:
                map_feat = self.null_mapper_embed.unsqueeze(0).expand(b, -1)
            cond_features.append(map_feat)

        all_features = torch.cat(cond_features, dim=-1)  # (B, num_cond_slots * dim_cond_fourier)
        c_global = self.cond_joint_mlp(all_features)  # (B, dim_h)

        c_global = c_global + self.time_mlp(t)
        c_global = c_global.unsqueeze(1)

        for mmdit_block in self.mmdit_blocks:
            x, a = mmdit_block(x, a, c_global, mask_x=mask_x, mask_a=mask_a)

        for dit_block in self.dit_blocks:
            x = dit_block(x, c_global, attn_mask=attn_mask_x)

        x = self.final(x, c_global)
        return x[:, :orig_x_len, :]
