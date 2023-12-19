import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import repeat
from modules.utils import right_pad_dims_to


@torch.jit.script
def beta_linear_log_snr(t: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.expm1(1e-4 + 10 * (t**2)))


@torch.jit.script
def alpha_cosine_log_snr(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    res = (torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1
    return -torch.log(res.clamp(min=1e-8))


def log_snr_to_alpha_sigma(log_snr: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


class GaussianDiffusionContinuousTimes(nn.Module):
    def __init__(
        self: "GaussianDiffusionContinuousTimes",
        noise_schedule: str = "cosine",
        timesteps: int = 1000,
    ) -> None:
        super().__init__()

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            msg = f"Unknown noise schedule: {noise_schedule}"
            raise ValueError(msg)

        self.timesteps = timesteps

    def get_times(
        self: "GaussianDiffusionContinuousTimes",
        batch_size: int,
        noise_level: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.full((batch_size,), noise_level, device=device, dtype=torch.float32)

    def sample_random_times(
        self: "GaussianDiffusionContinuousTimes",
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.zeros((batch_size,), device=device, dtype=torch.float32).uniform_(0, 1)

    def get_condition(self: "GaussianDiffusionContinuousTimes", t: torch.Tensor) -> torch.Tensor:
        return self.log_snr(t)

    def get_sampling_timesteps(
        self: "GaussianDiffusionContinuousTimes",
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        times = torch.linspace(1.0, 0.0, self.timesteps + 1, device=device, dtype=torch.float32)
        times = repeat(times, "t -> b t", b=batch_size)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def q_posterior(
        self: "GaussianDiffusionContinuousTimes",
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if t_next is None:
            t_next = (t - 1.0 / self.timesteps).clamp(min=0.0)

        log_snr = right_pad_dims_to(x_t, self.log_snr(t))
        log_snr_next = right_pad_dims_to(x_t, self.log_snr(t_next))

        alpha, _ = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        c = -torch.expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_0)

        posterior_variance = (sigma_next**2) * c
        posterior_log_variance = torch.log(posterior_variance.clamp(min=1e-20))
        return posterior_mean, posterior_variance, posterior_log_variance

    def q_sample(
        self: "GaussianDiffusionContinuousTimes",
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = x_0.dtype

        if isinstance(t, float):
            batch = x_0.shape[0]
            t = torch.full((batch,), t, dtype=dtype, device=x_0.device)

        noise = torch.randn_like(x_0) if noise is None else noise
        log_snr = self.log_snr(t).to(dtype=dtype)
        log_snr_padded = right_pad_dims_to(x_0, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_padded)

        return alpha * x_0 + sigma * noise, log_snr, alpha, sigma

    def predict_start_from_noise(
        self: "GaussianDiffusionContinuousTimes",
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        log_snr = right_pad_dims_to(x_t, self.log_snr(t))
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min=1e-8)
