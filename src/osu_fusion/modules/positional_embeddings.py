import math
import os

import torch
import torch.nn as nn
from torch.profiler import record_function

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self: "SinusoidalPositionEmbedding", dim: int, theta: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward_body(self: "SinusoidalPositionEmbedding", x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

    def forward(self: "SinusoidalPositionEmbedding", x: torch.Tensor) -> torch.Tensor:
        if DEBUG:
            with record_function("SinusoidalPositionEmbedding"):
                return self.forward_body(x)
        return self.forward_body(x)
