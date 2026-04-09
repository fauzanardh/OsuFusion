import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.profiler import record_function

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"


class RMSNorm(nn.Module):
    def __init__(
        self: "RMSNorm",
        dim: int,
    ) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim) * dim**0.5)

    def forward(self: "RMSNorm", x: torch.Tensor) -> torch.Tensor:
        if DEBUG:
            with record_function("RMSNorm"):
                return F.normalize(x, dim=-1) * self.g
        return F.normalize(x, dim=-1) * self.g


class MultiHeadRMSNorm(nn.Module):
    def __init__(
        self: "MultiHeadRMSNorm",
        dim: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(heads, 1, dim) * dim**0.5)

    def forward(self: "MultiHeadRMSNorm", x: torch.Tensor) -> torch.Tensor:
        if DEBUG:
            with record_function("MultiHeadRMSNorm"):
                return F.normalize(x, dim=-1) * self.g
        return F.normalize(x, dim=-1) * self.g
