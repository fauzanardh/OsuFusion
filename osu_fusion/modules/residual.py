from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class GlobalContext(nn.Module):
    """Attention-esque squeeze-excite module"""

    def __init__(self: "GlobalContext", dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.to_k = nn.Conv1d(dim_in, 1, 1)
        inner_dim = max(3, dim_out // 2)

        self.layers = nn.Sequential(
            nn.Conv1d(dim_in, inner_dim, 1),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim_out, 1),
            nn.Sigmoid(),
        )

    def forward(self: "GlobalContext", x: torch.Tensor) -> torch.Tensor:
        context = self.to_k(x)
        out = torch.einsum("b i d, b j d -> b i j", x, context.softmax(dim=-1))
        return self.layers(out)


class Block(nn.Module):
    def __init__(
        self: "Block",
        dim_in: int,
        dim_out: int,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(1, dim_out) if norm else nn.Identity()
        self.activation = nn.SiLU()

    def forward(self: "Block", x: torch.Tensor, scale_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self: "ResidualBlock",
        dim_in: int,
        dim_out: int,
        dim_cond: int,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_cond, dim_out * 2),
        )
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out, dim_out)

        self.res_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.gca = GlobalContext(dim_out, dim_out)

        self.gradient_checkpointing = False

    def forward_body(
        self: "ResidualBlock",
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        c = self.mlp(c)
        c = rearrange(c, "b d -> b d 1")
        scale_shift = c.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        h = h * self.gca(h)

        return h + self.res_conv(x)

    def forward(
        self: "ResidualBlock",
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_body, x, c, use_reentrant=True)
        else:
            return self.forward_body(x, c)
