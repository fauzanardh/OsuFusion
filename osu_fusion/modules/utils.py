from contextlib import contextmanager
from typing import Generator

import torch


# Classifier-free Guidance stuff
def prob_mask_like(shape: torch.Size, prob: float, device: torch.device) -> torch.Tensor:
    if prob == 0.0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    elif prob == 1.0:
        return torch.ones(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).uniform_(0.0, 1.0) < prob


# Used for profiling
@contextmanager
def dummy_context_manager() -> Generator[None, None, None]:
    yield
