from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from osu_fusion.library.osu.beatmap import Beatmap
from osu_fusion.library.osu.hit_objects import Slider, Spinner

ONSET_MIN_TIME = 4
ONSET_MAX_TIME = 11


def onsets(beatmap: Beatmap, frame_times: npt.NDArray) -> npt.NDArray:
    onsets_ = np.full_like(frame_times, 2**ONSET_MAX_TIME)
    for hit_object in beatmap.hit_objects:
        hit = frame_times - hit_object.t
        region = (hit >= 0) & (hit <= 2**ONSET_MAX_TIME)
        onsets_[region] = hit[region]
    log_onsets = np.log2(onsets_ + 2**ONSET_MIN_TIME).clip(ONSET_MIN_TIME, ONSET_MAX_TIME)
    return (log_onsets - ONSET_MIN_TIME) / (ONSET_MAX_TIME - ONSET_MIN_TIME)


def decode_onsets(onsets_: npt.NDArray) -> List[int]:
    return find_peaks(-onsets_, height=0.6, distance=4)[0].tolist()


Real = Union[int, float]


def combo_regions(beatmap: Beatmap) -> List[Tuple[Real, Real]]:
    new_combo_regions = []
    region_end = None
    for hit_object in beatmap.hit_objects[::-1]:
        if region_end is None:
            region_end = hit_object.end_time() + 1
        if hit_object.new_combo:
            new_combo_regions.insert(0, (hit_object.t, region_end))
            region_end = None
    return new_combo_regions


def extents(regions: List[Tuple[Real, Real]], frame_times: npt.NDArray) -> npt.NDArray:
    holds = np.zeros_like(frame_times)
    for s, e in regions:
        holds[(frame_times >= s) & (frame_times < e)] = 1
    return holds


def decode_extents(extents_: npt.NDArray) -> Tuple[List[int], List[int]]:
    before_below = extents_[:-1] <= 0
    after_below = extents_[1:] <= 0

    start_idxs = sorted(np.argwhere(before_below & ~after_below)[:, 0].tolist())
    end_idxs = sorted(np.argwhere(~before_below & after_below)[:, 0].tolist())

    cursor = 0
    for cursor, start in enumerate(start_idxs):
        try:
            while start >= end_idxs[cursor]:
                end_idxs.pop(cursor)
        except IndexError:
            break
    cursor += 1

    return start_idxs[:cursor], end_idxs[:cursor]


def hit_signals(beatmap: Beatmap, frame_times: npt.NDArray) -> npt.NDArray:
    signals = np.stack(
        [
            onsets(beatmap, frame_times),
            extents(
                [
                    (hit_object.t, hit_object.end_time())
                    for hit_object in beatmap.hit_objects
                    if isinstance(hit_object, (Slider, Spinner))
                ],
                frame_times,
            ),
            extents(
                [
                    (hit_object.t, hit_object.t + hit_object.slide_duration)
                    for hit_object in beatmap.hit_objects
                    if isinstance(hit_object, Slider)
                ],
                frame_times,
            ),
            extents(
                combo_regions(beatmap),
                frame_times,
            ),
        ],
    )

    return signals
