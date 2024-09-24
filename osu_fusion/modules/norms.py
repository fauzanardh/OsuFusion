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

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1.0)

    def forward(self: "RMSNorm", x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.g
