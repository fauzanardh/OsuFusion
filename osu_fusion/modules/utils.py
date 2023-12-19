import torch


def right_pad_dims_to(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))
