import torch

from osu_fusion.data.encode import SequenceEncoding


def flip_cursor_horizontal(x: torch.Tensor) -> torch.Tensor:
    x[:, SequenceEncoding.X] = -x[:, SequenceEncoding.X]
    return x


def flip_cursor_vertical(x: torch.Tensor) -> torch.Tensor:
    x[:, SequenceEncoding.Y] = -x[:, SequenceEncoding.Y]
    return x
