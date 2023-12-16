import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812
from tqdm.auto import tqdm

from library.osu.from_beatmap import TOTAL_DIM


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class BetaScheduler:
    def __init__(self: "BetaScheduler", betas: torch.Tensor, net: nn.Module) -> None:
        self._net = net

        self.betas = betas
        self.timesteps = len(betas)

        self.alphas = 1.0 - betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas_cumprod = self.alphas_cumprod.rsqrt()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()

        self.posterior_mean_x0_coef = self.betas * self.alphas_cumprod_prev.sqrt() / (1.0 - self.alphas_cumprod)
        self.posterior_mean_xt_coef = (
            (1.0 - self.alphas_cumprod_prev) * self.alphas.sqrt() / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        assert (self.posterior_variance[1:] != 0.0).all(), "variance is zero"

    def net(self: "BetaScheduler", x: torch.Tensor, a: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self._net(x, a, t, c)

    def q_sample(
        self: "BetaScheduler",
        x: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior_mean_var(self: "BetaScheduler", x0: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        posterior_mean_x0_coef_t = extract(self.posterior_mean_x0_coef, t, x.shape)
        posterior_mean_xt_coef_t = extract(self.posterior_mean_xt_coef, t, x.shape)

        posterior_mean = posterior_mean_x0_coef_t * x0 + posterior_mean_xt_coef_t * x
        posterior_variance = extract(self.posterior_variance, t, x.shape)

        return posterior_mean, posterior_variance

    def p_eps_mean_var(
        self: "BetaScheduler",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        model_eps = self._net(x, a, t, c)

        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)

        model_x0 = sqrt_recip_alphas_cumprod_t * x - model_eps * sqrt_recipm1_alphas_cumprod_t
        model_mean, model_variance = self.q_posterior_mean_var(model_x0, x, t)

        return model_eps, model_mean, model_variance, model_x0

    @torch.no_grad()
    def sample(
        self: "BetaScheduler",
        a: torch.Tensor,
        c: torch.Tensor,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, _, n = a.shape

        if x is None:
            x = torch.zeros((b, TOTAL_DIM, n), device=a.device)

        for i in tqdm(list(reversed(range(self.timesteps))), desc="Sampling from model"):
            t = torch.full((b,), i, device=a.device, dtype=torch.long)

            _, model_mean, model_variance, _ = self.p_eps_mean_var(x, a, t, c)
            x = model_mean if i == 0 else model_mean + model_variance.sqrt() * torch.randn_like(x)

        return x


class CosineBetaScheduler(BetaScheduler):
    def __init__(self: "CosineBetaScheduler", timesteps: int, net: nn.Module) -> None:
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)

        super().__init__(betas, net)


class StridedBetaScheduler(BetaScheduler):
    def __init__(
        self: "StridedBetaScheduler",
        scheduler: BetaScheduler,
        sample_steps: int,
        net: nn.Module,
        ddim: bool = True,
    ) -> None:
        use_timesteps = set(range(0, scheduler.timesteps, int(scheduler.timesteps / sample_steps) + 1))
        self.timesteps_map = []
        self.ddim = ddim

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alphas_cumprod in enumerate(scheduler.alphas_cumprod):
            if i in use_timesteps:
                self.timesteps_map.append(i)
                new_betas.append(1.0 - alphas_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alphas_cumprod

        super().__init__(torch.tensor(new_betas), net)

    def net(
        self: "StridedBetaScheduler",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.tensor(self.timesteps_map, device=t.device, dtype=t.dtype)[t]
        return self._net(x, a, t, c)

    def p_eps_mean_var(
        self: "StridedBetaScheduler",
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        model_eps, model_mean, model_variance, model_x0 = super().p_eps_mean_var(x, a, t, c)

        if self.ddim:
            eta = 0
            alpha_cumprod_prev_t = extract(self.alphas_cumprod_prev, t, x.shape)
            model_variance = eta**2 * model_variance
            model_mean = (
                model_x0 * alpha_cumprod_prev_t.sqrt()
                + model_eps * (1.0 - alpha_cumprod_prev_t - model_variance).sqrt()
            )

        return model_eps, model_mean, model_variance, model_x0
