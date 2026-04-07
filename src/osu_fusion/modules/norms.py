import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.profiler import record_function

from osu_fusion.modules.utils import dummy_context_manager

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"


class RMSNorm(nn.Module):
    def __init__(
        self: "RMSNorm",
        dim: int,
    ) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward_body(self: "RMSNorm", x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.g

    def forward(self: "RMSNorm", x: torch.Tensor) -> torch.Tensor:
        context_manager = record_function("RMSNorm") if DEBUG else dummy_context_manager()
        with context_manager:
            return self.forward_body(x)


class MultiHeadRMSNorm(nn.Module):
    def __init__(
        self: "MultiHeadRMSNorm",
        dim: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(heads, 1, dim))

    def forward_body(self: "MultiHeadRMSNorm", x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.g

    def forward(self: "MultiHeadRMSNorm", x: torch.Tensor) -> torch.Tensor:
        context_manager = record_function("MultiHeadRMSNorm") if DEBUG else dummy_context_manager()
        with context_manager:
            return self.forward_body(x)
