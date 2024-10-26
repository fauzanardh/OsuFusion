import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(
        self: "RMSNorm",
        dim: int,
    ) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self: "RMSNorm", x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.g


class MultiHeadRMSNorm(nn.Module):
    def __init__(
        self: "MultiHeadRMSNorm",
        dim: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self: "MultiHeadRMSNorm", x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.g
