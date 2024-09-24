from enum import IntEnum

import numpy as np
import numpy.typing as npt

from osu_fusion.data.cursor import cursor_signal
from osu_fusion.data.hit import hit_signals
from osu_fusion.osu.beatmap import Beatmap

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


def encode_beatmap(beatmap: Beatmap, frame_times: npt.NDArray) -> npt.NDArray:
    hit = hit_signals(beatmap, frame_times)
    cursor = cursor_signal(beatmap, frame_times) * 2 - 1

    return np.concatenate([hit, cursor], axis=0)
