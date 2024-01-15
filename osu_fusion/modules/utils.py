import torch
from einops import rearrange


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
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int = 0,
    scale: float = 1.0,
    seq_dim: int = -2,
) -> torch.Tensor:
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"Features dim {t.shape[-1]} must be larger than the positional encoding dim {rot_dim}"

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)
