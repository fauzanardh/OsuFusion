import torch
from torch.nn import functional as F

from osu_fusion.data.encode import BeatmapEncoding


# Flip cursor horizontally
def flip_cursor_horizontal(x: torch.Tensor) -> torch.Tensor:
    x[BeatmapEncoding.CURSOR_X] = -x[BeatmapEncoding.CURSOR_X]
    return x


# Flip cursor vertically
def flip_cursor_vertical(x: torch.Tensor) -> torch.Tensor:
    x[BeatmapEncoding.CURSOR_Y] = -x[BeatmapEncoding.CURSOR_Y]
    return x


# Simulate DT
def double_time(x: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(x.unsqueeze(0), scale_factor=1.5, mode="nearest").squeeze(0)
    return x


# Simulate HT
def half_time(x: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(x.unsqueeze(0), scale_factor=0.75, mode="nearest").squeeze(0)
    return x
