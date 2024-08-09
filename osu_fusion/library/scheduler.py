from typing import Dict, Union

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange


def log(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.log(x + eps)


class EDMScheduler:
    def __init__(
        self: "EDMScheduler",
        sigma_min: float = 0.002,
        sima_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        clamp: bool = True,
        clamp_value: float = 1.0,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sima_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.p_mean = p_mean
        self.p_std = p_std
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.clamp = clamp
        self.clamp_value = clamp_value

    def noise_distribution(self: "EDMScheduler", batch_size: int, device: torch.device) -> torch.Tensor:
        return (self.p_mean + self.p_std * torch.randn((batch_size,), device=device)).exp()

    def preconditioned_network_forward(
        self: "EDMScheduler",
        forward_fn: callable,
        x_noisy: torch.Tensor,
        a: torch.Tensor,
        sigma: Union[float, torch.Tensor],
        c: torch.Tensor,
        **kwargs: Dict,
    ) -> torch.Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.full((batch_size,), sigma, device=device)
        padded_sigma = rearrange(sigma, "b -> b 1 1")

        c_in = 1 * (padded_sigma**2 + self.sigma_data**2) ** -0.5
        c_noise = log(sigma) * 0.25
        c_skip = self.sigma_data**2 / (padded_sigma**2 + self.sigma_data**2)
        c_out = padded_sigma * self.sigma_data * (padded_sigma**2 + self.sigma_data**2) ** -0.5

        net_out = forward_fn(c_in * x_noisy, a, c_noise, c, **kwargs)
        out = c_skip * x_noisy + c_out * net_out

        if not self.clamp:
            return out
        return torch.clamp(out, -self.clamp_value, self.clamp_value)

    def sample_schedule(self: "EDMScheduler", num_sample_timesteps: int, device: torch.device) -> torch.Tensor:
        inv_rho = 1.0 / self.rho

        steps = torch.arange(num_sample_timesteps, dtype=torch.float32, device=device)
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (num_sample_timesteps - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho
        sigmas = F.pad(sigmas, (1, 0), value=0.0)  # prepend 0.0 to sigmas
        return sigmas
