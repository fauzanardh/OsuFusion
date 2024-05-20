from enum import IntEnum

import numpy as np
import numpy.typing as npt

from osu_fusion.library.osu.beatmap import Beatmap
from osu_fusion.library.osu.data.cursor import cursor_signal
from osu_fusion.library.osu.data.hit import hit_signals

BeatmapEncoding = IntEnum(
    "BeatmapEncoding",
    [
        # hit signals
        "HIT",
        "SUSTAIN",
        "SLIDER",
        "COMBO",
        # cursor signals
        "CURSOR_X",
        "CURSOR_Y",
    ],
    start=0,
)
TOTAL_DIM = len(BeatmapEncoding)


def encode_beatmap(beatmap: Beatmap, frame_times: npt.NDArray) -> npt.NDArray:
    hit = hit_signals(beatmap, frame_times)
    cursor = cursor_signal(beatmap, frame_times)

    return np.concatenate([hit, cursor], axis=0) * 2 - 1
