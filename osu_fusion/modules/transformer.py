import torch
import torch.nn as nn
from einops import rearrange

from osu_fusion.modules.attention import MultiHeadAttention


class FeedForward(nn.Sequential):
    def __init__(
        self: "FeedForward",
        dim: int,
        dim_mult: int = 4,
    ) -> None:
        inner_dim = dim * dim_mult
        super().__init__(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, dim, bias=False),
        )


class TransformerBlock(nn.Module):
    def __init__(
        self: "TransformerBlock",
        dim: int,
        dim_context: int,
        dim_head: int = 32,
        heads: int = 8,
        dropout: float = 0.1,
        sdpa: bool = True,
        use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            dim,
            dim_head=dim_head,
            heads=heads,
            dropout=dropout,
            sdpa=sdpa,
            use_rotary_emb=use_rotary_emb,
        )
        self.feed_forward = FeedForward(dim)
        self.cross_attention = MultiHeadAttention(
            dim,
            dim_context=dim_context,
            dim_head=dim_head,
            heads=heads,
            dropout=dropout,
            sdpa=sdpa,
            is_cross_attention=True,
            use_rotary_emb=use_rotary_emb,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self: "TransformerBlock", x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.norm1(x))
        x = x + self.cross_attention(self.norm2(x), context)
        x = x + self.feed_forward(self.norm3(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self: "Transformer",
        dim: int,
        dim_context: int,
        dim_head: int = 32,
        heads: int = 8,
        depth: int = 4,
        dropout: float = 0.1,
        sdpa: bool = True,
        use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    dim_context,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=dropout,
                    sdpa=sdpa,
                    use_rotary_emb=use_rotary_emb,
                )
                for _ in range(depth)
            ],
        )

    def forward(self: "Transformer", x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b d n -> b n d")
        for layer in self.layers:
            x = layer(x, context)
        x = rearrange(x, "b n d -> b d n")
        return x
