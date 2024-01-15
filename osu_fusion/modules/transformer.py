import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from osu_fusion.modules.attention import MultiHeadAttention


class RMSNorm(nn.Module):
    def __init__(self: "RMSNorm", dim: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self: "RMSNorm", x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class FeedForward(nn.Sequential):
    def __init__(
        self: "FeedForward",
        dim: int,
        dim_mult: int = 4,
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
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)

        self.gradient_checkpointing = False

    def forward_body(self: "TransformerBlock", x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.norm1(x))
        x = x + self.cross_attention(self.norm2(x), context)
        x = x + self.feed_forward(self.norm3(x))
        return x

    def forward(self: "TransformerBlock", x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, context, use_reentrant=True)
        else:
            return self.forward_body(x, context)


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
        for layer in self.layers:
            x = layer(x, context)
        return x
