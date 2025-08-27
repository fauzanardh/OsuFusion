from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
from slider.beatmap import Beatmap, Slider
from slider.curve import Linear, MultiBezier, Perfect

from osu_fusion.data.enum import BeatmapEncoding


def get_path_arc_lengths(path_points: npt.NDArray) -> Tuple[npt.NDArray, float]:
    """
    Calculates the cumulative arc length along a path defined by points.
    This is used for calculating anchor timings in Linear sliders.
    """
    if len(path_points) < 2:
        return np.array([0.0]), 0.0

    segment_lengths = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    total_length = cumulative_lengths[-1]

    return cumulative_lengths, total_length


def flips(frame_times: npt.NDArray, event_times: List[float]) -> npt.NDArray:
    signal = np.full_like(frame_times, 0.0)
    current_state = 0.0
    for t in event_times:
        frame_idx = np.searchsorted(frame_times, t)
        if frame_idx < len(frame_times):
            current_state = 1.0 - current_state
            signal[frame_idx:] = current_state
    return signal


def decode_flips(flips_: npt.NDArray) -> List[int]:
    return np.where(np.diff(flips_) != 0)[0].tolist()


Real = Union[int, float]


def combo_regions(beatmap: Beatmap) -> List[Tuple[Real, Real]]:
    new_combo_regions = []
    region_end = None
    for hit_object in beatmap.hit_objects()[::-1]:
        if region_end is None:
            region_end = hit_object.end_time.total_seconds() * 1000 + 1
        if hit_object.new_combo:
            new_combo_regions.insert(0, (hit_object.time.total_seconds() * 1000, region_end))
            region_end = None
    return new_combo_regions


def extents(frame_times: npt.NDArray, regions: List[Tuple[float, float]]) -> npt.NDArray:
    signal = np.zeros_like(frame_times)
    for start, end in regions:
        signal[(frame_times >= start) & (frame_times < end)] = 1
    return signal


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


def hit_signals(beatmap: Beatmap, frame_times: npt.NDArray) -> npt.NDArray:  # noqa: C901
    hit_objects = beatmap.hit_objects()
    signals = np.zeros((len(BeatmapEncoding) - 2, len(frame_times)), dtype=np.float32)
    signals[BeatmapEncoding.HIT] = flips(frame_times, [ho.time.total_seconds() * 1000 for ho in hit_objects])
    signals[BeatmapEncoding.SUSTAIN] = extents(
        frame_times,
        [
            (ho.time.total_seconds() * 1000, ho.end_time.total_seconds() * 1000)
            for ho in hit_objects
            if isinstance(ho, Slider)
        ],
    )
    signals[BeatmapEncoding.SLIDER] = extents(
        frame_times,
        [
            (
                ho.time.total_seconds() * 1000,
                ho.time.total_seconds() * 1000
                + (ho.end_time.total_seconds() - ho.time.total_seconds()) * 1000 / ho.repeat,
            )
            for ho in hit_objects
            if isinstance(ho, Slider)
        ],
    )
    linear_anchors, perfect_anchors = [], []
    white_bezier_anchors, red_bezier_anchors = [], []
    for hit_object in hit_objects:
        if not isinstance(hit_object, Slider):
            continue

        total_slider_length = hit_object.length
        if total_slider_length <= 1e-6:
            continue

        slide_duration = (
            (hit_object.end_time.total_seconds() - hit_object.time.total_seconds()) * 1000 / hit_object.repeat
        )
        start_time = hit_object.time.total_seconds() * 1000

        if isinstance(hit_object.curve, Perfect):
            anchor_time = start_time + 0.5 * slide_duration
            perfect_anchors.append(anchor_time)

        elif isinstance(hit_object.curve, Linear):
            control_points = np.array(hit_object.curve.points)
            if len(control_points) <= 2:
                continue
            anchor_distances, _ = get_path_arc_lengths(control_points)
            for i in range(1, len(control_points) - 1):
                dist_to_anchor = anchor_distances[i]
                time_proportion = dist_to_anchor / total_slider_length
                anchor_time = start_time + time_proportion * slide_duration
                linear_anchors.append(anchor_time)

        elif isinstance(hit_object.curve, MultiBezier):
            control_points = hit_object.curve.points
            if len(control_points) < 2:
                continue

            path_points = np.array(hit_object.curve.points)
            if len(path_points) <= 2:
                continue

            cumulative_lengths, total_length = get_path_arc_lengths(path_points)
            if total_length <= 1e-6:
                continue

            for i in range(1, len(path_points)):
                dist_proportion = cumulative_lengths[i] / total_length
                anchor_time = start_time + dist_proportion * slide_duration

                is_red_anchor = (i + 1 < len(path_points)) and np.array_equal(path_points[i], path_points[i + 1])

                if is_red_anchor:
                    red_bezier_anchors.append(anchor_time)
                elif not np.array_equal(path_points[i], path_points[i - 1]):
                    white_bezier_anchors.append(anchor_time)

    signals[BeatmapEncoding.WHITE_BEZIER_ANCHORS] = flips(frame_times, sorted(white_bezier_anchors))
    signals[BeatmapEncoding.RED_BEZIER_ANCHORS] = flips(frame_times, sorted(red_bezier_anchors))
    signals[BeatmapEncoding.LINEAR_ANCHORS] = flips(frame_times, sorted(linear_anchors))
    signals[BeatmapEncoding.PERFECT_ANCHORS] = flips(frame_times, sorted(perfect_anchors))
    signals[BeatmapEncoding.COMBO] = flips(
        frame_times,
        [ho.time.total_seconds() * 1000 for ho in hit_objects if ho.new_combo],
    )
    kiai_regions = []
    for i, tp in enumerate(beatmap.timing_points):
        if tp.kiai_mode:
            start_time = tp.offset.total_seconds() * 1000
            end_time = (
                beatmap.timing_points[i + 1].offset.total_seconds() * 1000
                if i + 1 < len(beatmap.timing_points)
                else frame_times[-1]
            )
            kiai_regions.append((start_time, end_time))
    signals[BeatmapEncoding.KIAI] = extents(frame_times, kiai_regions)

    return signals
