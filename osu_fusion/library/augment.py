import torch

from osu_fusion.library.osu.data.encode import BeatmapEncoding


# Flip cursor horizontally
def flip_cursor_horizontal(x: torch.Tensor) -> torch.Tensor:
    x[BeatmapEncoding.CURSOR_X] = -x[BeatmapEncoding.CURSOR_X]
    return x


# Flip cursor vertically
def flip_cursor_vertical(x: torch.Tensor) -> torch.Tensor:
    x[BeatmapEncoding.CURSOR_Y] = -x[BeatmapEncoding.CURSOR_Y]
    return x


# Randomly add maximum of `pixel` noise to cursor position
def add_cursor_noise(x: torch.Tensor, pixel: int = 2) -> torch.Tensor:
    x[BeatmapEncoding.CURSOR_X] += torch.randn_like(x[BeatmapEncoding.CURSOR_X]) * (pixel / 512)
    x[BeatmapEncoding.CURSOR_Y] += torch.randn_like(x[BeatmapEncoding.CURSOR_Y]) * (pixel / 384)
    return x
