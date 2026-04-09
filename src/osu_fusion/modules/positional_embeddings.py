import math
import os

import torch
import torch.nn as nn
from einops import rearrange
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


class LearnedSinusoidalPosEmb(nn.Module):
    """
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8
    """

    def __init__(self: "LearnedSinusoidalPosEmb", dim: int) -> None:
        super().__init__()
        assert (dim % 2) == 0, "dim must be even"
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward_body(self: "LearnedSinusoidalPosEmb", x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim=-1)

    def forward(self: "LearnedSinusoidalPosEmb", x: torch.Tensor) -> torch.Tensor:
        if DEBUG:
            with record_function("LearnedSinusoidalPosEmb"):
                return self.forward_body(x)
        return self.forward_body(x)
