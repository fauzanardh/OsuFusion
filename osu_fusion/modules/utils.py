import torch


def right_pad_dims_to(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


# Classifier-free Guidance stuff
def prob_mask_like(shape: torch.Size, prob: float, device: torch.device) -> torch.Tensor:
    if prob == 0.0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    elif prob == 1.0:
        return torch.ones(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).uniform_(0.0, 1.0) < prob


# Rotary Position Embedding stuff
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[:, :, :, : x.shape[-1]]
    sin = sin[:, :, :, : x.shape[-1]]

    return (x * cos) + (rotate_half(x) * sin)
