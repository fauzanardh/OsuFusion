from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


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
        dim_time: Optional[int] = None,
        dim_cond: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.has_time_cond = dim_time is not None
        self.has_cond = dim_cond is not None

        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(int(dim_time) + int(dim_cond), dim_out * 2),
            )
            if dim_time or dim_cond
            else None
        )
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out, dim_out)

        self.res_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(
        self: "ResidualBlock",
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and (self.has_time_cond or self.has_cond):
            cond_emb = tuple(filter(lambda tensor: tensor is not None, (t, c)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, "b c -> b c 1")
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)
