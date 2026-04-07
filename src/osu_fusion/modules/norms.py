import os

import torch
import torch.nn as nn
from torch.profiler import record_function

from osu_fusion.modules.triton_kernels import fused_rms_norm

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"


class RMSNorm(nn.Module):
    def __init__(
        self: "RMSNorm",
        dim: int,
    ) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self: "RMSNorm", x: torch.Tensor) -> torch.Tensor:
        if DEBUG:
            with record_function("RMSNorm"):
                return fused_rms_norm(x, self.g, self.scale)
        return fused_rms_norm(x, self.g, self.scale)


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
        if DEBUG:
            with record_function("MultiHeadRMSNorm"):
                return fused_rms_norm(x, self.g, self.scale)
        return fused_rms_norm(x, self.g, self.scale)
