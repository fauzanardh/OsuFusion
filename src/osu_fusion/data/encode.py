import numpy as np
import numpy.typing as npt

from osu_fusion.data.cursor import cursor_signal
from osu_fusion.data.hit import hit_signals
from osu_fusion.osu.beatmap import Beatmap


def encode_beatmap(beatmap: Beatmap, frame_times: npt.NDArray) -> npt.NDArray:
    hit = hit_signals(beatmap, frame_times)
    cursor = cursor_signal(beatmap, frame_times)

    return np.concatenate([hit, cursor], axis=0) * 2 - 1
