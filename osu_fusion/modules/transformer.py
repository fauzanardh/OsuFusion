import torch
import torch.nn as nn

from osu_fusion.modules.attention import MultiHeadAttention


class FeedForward(nn.Sequential):
    def __init__(
        self: "FeedForward",
        dim: int,
        dim_mult: int = 2,
        dropout: float = 0.1,
    ) -> None:
        inner_dim = dim * dim_mult
        super().__init__(
            nn.Conv1d(dim, inner_dim, 1),
            nn.Conv1d(inner_dim, inner_dim, 3, padding=1, bias=True, groups=inner_dim),
            nn.GELU(),
            nn.Conv1d(inner_dim, dim, 1),
            nn.Dropout(dropout),
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
        linear: bool = False,
        use_rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            dim,
            dim_head=dim_head,
            heads=heads,
            dropout=dropout,
            linear=linear,
            use_rotary_emb=use_rotary_emb,
        )
        self.feed_forward = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)

        self.gradient_checkpointing = False

    def forward_body(self: "TransformerBlock", x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.norm1(x))
        x = x + self.feed_forward(x)
        return x

    def forward(self: "TransformerBlock", x: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, use_reentrant=True)
        else:
            return self.forward_body(x)


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
        linear: bool = False,
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
                    linear=linear,
                    use_rotary_emb=use_rotary_emb,
                )
                for _ in range(depth)
            ],
        )

    def forward(self: "Transformer", x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, context)
        return x
