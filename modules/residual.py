from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange

from modules.causal_convolution import CausalConv1d


class SqueezeExcite(nn.Module):
    def __init__(self: "SqueezeExcite", dim: int, reduction_factor: int = 4, dim_minimum: int = 8) -> None:
        super().__init__()
        hidden_dim = max(dim // reduction_factor, dim_minimum)
        self.layers = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self: "SqueezeExcite", x: torch.Tensor) -> torch.Tensor:
        seq, device = x.shape[-2], x.device

        cum_sum = x.cumsum(dim=-2)
        denominator = torch.arange(1, seq + 1, device=device).float()
        mean = cum_sum / rearrange(denominator, "n -> n 1")

        return x * self.layers(mean)


class Residual(nn.Module):
    def __init__(self: "Residual", fn: callable) -> None:
        super().__init__()
        self.fn = fn

    def forward(self: "Residual", x: torch.Tensor, **kwargs: Dict) -> torch.Tensor:
        return x + self.fn(x, **kwargs)


class ResidualBlock(nn.Module):
    def __init__(
        self: "ResidualBlock",
        dim_in: int,
        dim_out: int,
        dilation: int,
        kernel_size: int = 7,
        squeeze_excite: bool = True,
    ) -> None:
        super().__init__()
        self.layers = Residual(
            nn.Sequential(
                CausalConv1d(dim_in, dim_out, kernel_size, dilation=dilation),
                nn.ReLU(inplace=True),
                CausalConv1d(dim_out, dim_out, 1),
                nn.ReLU(inplace=True),
                SqueezeExcite(dim_out) if squeeze_excite else nn.Identity(),
            ),
        )

    def forward(self: "ResidualBlock", x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
