from copy import deepcopy
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def sigma_rel_to_gamma(sigma_rel: float) -> float:
    t = sigma_rel**-2
    gamma = np.roots([1, 7, 16 - t, 12 - t]).real.max()
    return gamma


class EMA(nn.Module):
    def __init__(
        self: "EMA",
        model: nn.Module,
        gamma: float = 6.94,
        sigma_rel: Optional[float] = None,
        update_after_step: int = 100,
        update_every: int = 10,
    ) -> None:
        super().__init__()

        # Karrass' EMA parameters
        self.gamma = gamma if sigma_rel is None else sigma_rel_to_gamma(sigma_rel)

        # EMA parameters
        self.update_after_step = update_after_step
        self.update_every = update_every

        self.online_model = [model]  # hack so pytorch doesn't include model in state_dict
        self.ema_model = deepcopy(model)
        self.ema_model.requires_grad_(False)

        self.parameter_names = {
            name
            for name, param in self.ema_model.named_parameters()
            if param.dtype in (torch.float32, torch.float16, torch.bfloat16)
        }
        self.buffer_names = {
            name
            for name, buffer in self.ema_model.named_buffers()
            if buffer.dtype in (torch.float32, torch.float16, torch.bfloat16)
        }

        self.register_buffer("initted", torch.tensor(False))
        self.register_buffer("step", torch.tensor(0))

    @property
    def model(self: "EMA") -> nn.Module:
        return self.online_model[0]

    @property
    def beta(self: "EMA") -> float:
        return (1 - 1 / (self.step + 1)) ** (1 + self.gamma)

    def get_params_iter(self: "EMA", model: nn.Module) -> Generator[Tuple[str, torch.Tensor], None, None]:
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self: "EMA", model: nn.Module) -> Generator[Tuple[str, torch.Tensor], None, None]:
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self: "EMA") -> None:
        for (_, param), (_, ema_param) in zip(
            self.get_params_iter(self.model),
            self.get_params_iter(self.ema_model),
        ):
            ema_param.copy_(param)

        for (_, buffer), (_, ema_buffer) in zip(
            self.get_buffers_iter(self.model),
            self.get_buffers_iter(self.ema_model),
        ):
            ema_buffer.copy_(buffer)

    def copy_params_from_ema_to_model(self: "EMA") -> None:
        for (_, param), (_, ema_param) in zip(
            self.get_params_iter(self.model),
            self.get_params_iter(self.ema_model),
        ):
            param.copy_(ema_param)

        for (_, buffer), (_, ema_buffer) in zip(
            self.get_buffers_iter(self.model),
            self.get_buffers_iter(self.ema_model),
        ):
            buffer.copy_(ema_buffer)

    def update(self: "EMA") -> None:
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step < self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.fill_(True)

        self.update_moving_average()

    @torch.no_grad()
    def update_moving_average(self: "EMA") -> None:
        current_decay = self.beta

        for (_, param), (_, ema_param) in zip(
            self.get_params_iter(self.model),
            self.get_params_iter(self.ema_model),
        ):
            ema_param.lerp_(param, 1 - current_decay)

        for (_, buffer), (_, ema_buffer) in zip(
            self.get_buffers_iter(self.model),
            self.get_buffers_iter(self.ema_model),
        ):
            ema_buffer.lerp_(buffer, 1 - current_decay)

    def __call__(self: "EMA", *args: List, **kwargs: Dict) -> torch.Tensor:
        return self.ema_model(*args, **kwargs)
