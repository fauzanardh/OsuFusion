import torch
import torch.nn as nn

from osu_fusion.modules.attention import MultiHeadAttention


class FeedForward(nn.Sequential):
    def __init__(
        self: "FeedForward",
        dim: int,
        dim_mult: int = 2,
    ) -> None:
        inner_dim = dim * dim_mult
        super().__init__(
            nn.GroupNorm(1, dim),
            nn.Conv1d(dim, inner_dim, 1, bias=False),
            nn.GELU(),
            nn.GroupNorm(1, inner_dim),
            nn.Conv1d(inner_dim, dim, 1, bias=False),
        )


class TransformerBlock(nn.Module):
    def __init__(
        self: "TransformerBlock",
        dim: int,
        dim_head: int = 32,
        heads: int = 8,
        sdpa: bool = True,
        linear: bool = False,
        use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            dim,
            dim_head=dim_head,
            heads=heads,
            sdpa=sdpa,
            linear=linear,
            use_rotary_emb=use_rotary_emb,
        )
        self.feed_forward = FeedForward(dim)

        self.gradient_checkpointing = False

    def forward_body(self: "TransformerBlock", x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(x)
        x = x + self.feed_forward(x)
        return x

    def forward(self: "TransformerBlock", x: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, use_reentrant=True)
        else:
            return self.forward_body(x)
