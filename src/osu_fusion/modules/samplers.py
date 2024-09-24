import os

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from scipy import signal
from torch.nn import functional as F
from torch.profiler import record_function

from osu_fusion.modules.utils import dummy_context_manager

DEBUG = os.environ.get("DEBUG", False)


# Uses Sinc Kaiser Windowed Filter
class Upsample(nn.Module):
    def __init__(self: "Upsample", dim_in: int, dim_out: int, kernel_size: int = 17) -> None:
        super().__init__()
        self.scale_factor = 2
        self.conv = nn.Conv1d(dim_in, dim_out, 1, bias=False)

        kernel = self._create_sinc_kaiser_kernel(kernel_size)
        self.register_buffer("kernel", kernel)

    def _create_sinc_kaiser_kernel(self: "Upsample", kernel_size: int) -> torch.Tensor:
        width = 1 / self.scale_factor
        atten = signal.kaiser_atten(kernel_size, width)
        beta = signal.kaiser_beta(atten)

        t = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size // 2)
        sinc_func = torch.sinc(t / self.scale_factor)

        kaiser_window = torch.tensor(np.kaiser(kernel_size, beta), dtype=torch.float32)
        kernel = sinc_func * kaiser_window
        kernel = kernel / kernel.sum()
        return rearrange(kernel, "n -> 1 1 n")

    def forward_body(self: "Upsample", x: torch.Tensor) -> torch.Tensor:
        b, d, n = x.shape
        up_n = n * self.scale_factor
        x_upsampled = torch.zeros(b, d, up_n, device=x.device, dtype=x.dtype)
        x_upsampled[:, :, :: self.scale_factor] = x

        padding = self.kernel.shape[-1] // 2
        x_padded = F.pad(x_upsampled, (padding, padding), mode="reflect")
        x_filtered = F.conv1d(x_padded, repeat(self.kernel, "1 1 n -> d 1 n", d=d), groups=d)

        return self.conv(x_filtered)

    def forward(self: "Upsample", x: torch.Tensor) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("Upsample")
        with context_manager:
            return self.forward_body(x)


class Downsample(nn.Module):
    def __init__(self: "Downsample", dim_in: int, dim_out: int) -> None:
        super().__init__()
        # Asymmetric padding
        self.conv = nn.Conv1d(dim_in, dim_out, 7, stride=2, padding=0)

    def forward_body(self: "Downsample", x: torch.Tensor) -> torch.Tensor:
        pad = (0, 5)
        x = F.pad(x, pad=pad, mode="reflect")
        x = self.conv(x)
        return x

    def forward(self: "Downsample", x: torch.Tensor) -> torch.Tensor:
        context_manager = dummy_context_manager() if DEBUG else record_function("Downsample")
        with context_manager:
            return self.forward_body(x)
