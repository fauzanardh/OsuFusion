from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt

from osu_fusion.osu.beatmap import Beatmap
from osu_fusion.osu.hit_objects import Slider, Spinner


def flips(beatmap: Beatmap, frame_times: npt.NDArray, combo: bool = False) -> npt.NDArray:
    hit = np.full_like(frame_times, 0.0)
    current_state = 0.0
    for hit_object in beatmap.hit_objects:
        if not combo or hit_object.new_combo:
            closest_frame_idx = np.searchsorted(frame_times, hit_object.t)
            if closest_frame_idx < len(frame_times):
                current_state = 1.0 - current_state
                hit[closest_frame_idx:] = current_state
    return hit


def decode_flips(flips_: npt.NDArray) -> List[int]:
    return np.where(np.diff(flips_) != 0)[0].tolist()


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
    before_below = extents_[:-1] <= 0.5
    after_below = extents_[1:] <= 0.5

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
            flips(beatmap, frame_times),
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
            flips(beatmap, frame_times, combo=True),
        ],
    )

    return signals
