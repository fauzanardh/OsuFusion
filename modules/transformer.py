import torch
import torch.nn as nn
from einops import rearrange, repeat

from modules.attention import MultiHeadAttention


class FeedForward(nn.Sequential):
    def __init__(
        self: "FeedForward",
        dim: int,
        dim_mult: int = 4,
        dropout: float = 0.25,
    ) -> None:
        inner_dim = dim * dim_mult
        super().__init__(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.ReLU(),
            nn.LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )


class TransformerBlock(nn.Module):
    def __init__(
        self: "TransformerBlock",
        dim: int,
        dim_context: int,
        dim_head: int = 32,
        heads: int = 8,
        dropout: float = 0.25,
        sdpa: bool = True,
        ff_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            dim,
            dim_head=dim_head,
            heads=heads,
            dropout=dropout,
            sdpa=sdpa,
        )
        self.feed_forward = FeedForward(dim, dropout=ff_dropout)
        self.cross_attention = MultiHeadAttention(
            dim,
            dim_context=dim_context,
            dim_head=dim_head,
            heads=heads,
            dropout=dropout,
            sdpa=sdpa,
            is_cross_attention=True,
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
        dropout: float = 0.25,
        sdpa: bool = True,
        ff_dropout: float = 0.25,
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
                    ff_dropout=ff_dropout,
                )
                for _ in range(depth)
            ],
        )

    def forward(self: "Transformer", x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b d n -> b n d")
        for layer in self.layers:
            c = repeat(context, "b d -> b n d", n=x.shape[1])
            x = layer(x, c)
        x = rearrange(x, "b n d -> b d n")
        return x
