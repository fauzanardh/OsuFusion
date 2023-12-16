import math

import torch


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half_dim = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(
        timesteps.device,
    )
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
