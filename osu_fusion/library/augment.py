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
