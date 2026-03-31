from enum import IntEnum
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
from slider.beatmap import Beatmap, Circle, Slider, Spinner
from slider.curve import Catmull, Perfect

from osu_fusion.data.const import MS_PER_FRAME


class SequenceEncoding(IntEnum):
    IS_NOTE = 0
    OFFSET = 1
    X = 2
    Y = 3
    NEW_COMBO = 4
    IS_SLIDER_BODY = 5
    IS_SPINNER = 6
    SLIDER_LENGTH = 7
    SLIDER_REPEATS = 8

    TYPE_CIRCLE = 9
    TYPE_SLIDER_HEAD = 10
    TYPE_BEZIER_ANCHOR = 11
    TYPE_PERFECT_ANCHOR = 12
    TYPE_CATMULL_ANCHOR = 13
    TYPE_RED_ANCHOR = 14
    TYPE_LAST_ANCHOR = 15
    TYPE_SLIDER_END = 16
    TYPE_SPINNER = 17
    TYPE_SPINNER_END = 18


SEQ_DIM = len(SequenceEncoding)
LOG_SCALE_LENGTH = np.log1p(1000.0)
LOG_SCALE_REPEATS = np.log1p(10.0)

MAX_COLLISION_SEARCH = 16


def get_anchor_type(curve: Any, prev_point: Any, point: Any) -> int:  # noqa: ANN401
    if prev_point is not None and (point.x, point.y) == (prev_point.x, prev_point.y):
        return SequenceEncoding.TYPE_RED_ANCHOR
    if isinstance(curve, Perfect):
        return SequenceEncoding.TYPE_PERFECT_ANCHOR
    elif isinstance(curve, Catmull):
        return SequenceEncoding.TYPE_CATMULL_ANCHOR
    else:
        return SequenceEncoding.TYPE_BEZIER_ANCHOR


def get_path_arc_lengths(path_points: list) -> Tuple[npt.NDArray, float]:
    if len(path_points) < 2:
        return np.array([0.0]), 0.0
    arr = np.array([[p.x, p.y] for p in path_points])
    segment_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    return cumulative_lengths, float(cumulative_lengths[-1])


def encode_sequence(beatmap: Beatmap, total_frames: int) -> npt.NDArray:  # noqa: C901
    grid = np.full((total_frames, SEQ_DIM), -1.0, dtype=np.float32)

    hit_objects = beatmap.hit_objects()
    if not hit_objects:
        return grid

    first_ho = hit_objects[0]
    grid[:, SequenceEncoding.X] = (first_ho.position.x / 256.0) - 1.0
    grid[:, SequenceEncoding.Y] = (first_ho.position.y / 192.0) - 1.0

    def add_impulse(
        time_ms: float,
        x: float,
        y: float,
        new_combo: bool,
        length: float,
        repeats: int,
        event_type: int,
    ) -> None:
        frame_idx = int(time_ms // MS_PER_FRAME)
        if frame_idx >= total_frames:
            return

        # Expanding search for a free slot to avoid overwriting existing events
        if grid[frame_idx, SequenceEncoding.IS_NOTE] > 0.0:
            found = False
            for delta in range(1, MAX_COLLISION_SEARCH):
                candidate_fwd = frame_idx + delta
                if candidate_fwd < total_frames and grid[candidate_fwd, SequenceEncoding.IS_NOTE] <= 0.0:
                    frame_idx = candidate_fwd
                    found = True
                    break
                candidate_bwd = frame_idx - delta
                if candidate_bwd >= 0 and grid[candidate_bwd, SequenceEncoding.IS_NOTE] <= 0.0:
                    frame_idx = candidate_bwd
                    found = True
                    break
            if not found:
                return  # No free slot found, skip this event rather than corrupt

        # Compute offset relative to the actual frame used, preserving original time
        offset = (time_ms / MS_PER_FRAME) - frame_idx
        offset = float(np.clip(offset, 0.0, 1.0))

        norm_x = (x / 256.0) - 1.0
        norm_y = (y / 192.0) - 1.0

        grid[frame_idx:, SequenceEncoding.X] = norm_x
        grid[frame_idx:, SequenceEncoding.Y] = norm_y

        grid[frame_idx, SequenceEncoding.IS_NOTE] = 1.0
        grid[frame_idx, SequenceEncoding.OFFSET] = (offset * 2.0) - 1.0

        if new_combo:
            grid[frame_idx, SequenceEncoding.NEW_COMBO] = 1.0

        grid[frame_idx, SequenceEncoding.SLIDER_LENGTH] = (np.log1p(max(0.0, length)) / LOG_SCALE_LENGTH) * 2.0 - 1.0
        grid[frame_idx, SequenceEncoding.SLIDER_REPEATS] = (np.log1p(max(0.0, repeats)) / LOG_SCALE_REPEATS) * 2.0 - 1.0

        grid[frame_idx, 9:19] = -1.0
        grid[frame_idx, event_type] = 1.0

    for ho in hit_objects:
        time_ms = ho.time.total_seconds() * 1000.0

        if isinstance(ho, Circle):
            add_impulse(time_ms, ho.position.x, ho.position.y, ho.new_combo, 0, 0, SequenceEncoding.TYPE_CIRCLE)

        elif isinstance(ho, Spinner):
            add_impulse(time_ms, 256, 192, ho.new_combo, 0, 0, SequenceEncoding.TYPE_SPINNER)
            end_time_ms = ho.end_time.total_seconds() * 1000.0
            add_impulse(end_time_ms, 256, 192, False, 0, 0, SequenceEncoding.TYPE_SPINNER_END)

            start_idx = int(time_ms // MS_PER_FRAME)
            end_idx = min(int(end_time_ms // MS_PER_FRAME), total_frames - 1)
            if end_idx >= start_idx:
                grid[start_idx : end_idx + 1, SequenceEncoding.IS_SPINNER] = 1.0

        elif isinstance(ho, Slider):
            head_time = time_ms
            add_impulse(
                head_time,
                ho.position.x,
                ho.position.y,
                ho.new_combo,
                ho.length,
                ho.repeat,
                SequenceEncoding.TYPE_SLIDER_HEAD,
            )

            points = ho.curve.points
            slide_duration = ((ho.end_time.total_seconds() * 1000.0) - head_time) / ho.repeat

            if len(points) > 2:
                cumulative_lengths, total_length = get_path_arc_lengths(points)
                for i in range(1, len(points) - 1):
                    anchor_type = get_anchor_type(ho.curve, points[i - 1], points[i])
                    time_proportion = cumulative_lengths[i] / total_length if total_length > 1e-6 else 0.0
                    anchor_time = head_time + (time_proportion * slide_duration)
                    add_impulse(anchor_time, points[i].x, points[i].y, False, 0, 0, anchor_type)

            if len(points) > 1:
                cumulative_lengths, total_length = get_path_arc_lengths(points)
                time_proportion = cumulative_lengths[-1] / total_length if total_length > 1e-6 else 1.0
                last_anchor_time = head_time + (time_proportion * slide_duration)
                add_impulse(
                    last_anchor_time,
                    points[-1].x,
                    points[-1].y,
                    False,
                    0,
                    0,
                    SequenceEncoding.TYPE_LAST_ANCHOR,
                )

            end_time_ms = ho.end_time.total_seconds() * 1000.0
            end_pos = ho.curve(1.0) if ho.repeat % 2 != 0 else ho.position
            add_impulse(
                end_time_ms,
                end_pos.x,
                end_pos.y,
                False,
                ho.length,
                ho.repeat,
                SequenceEncoding.TYPE_SLIDER_END,
            )

            start_idx = int(head_time // MS_PER_FRAME)
            end_idx = min(int(end_time_ms // MS_PER_FRAME), total_frames - 1)
            if end_idx >= start_idx:
                grid[start_idx : end_idx + 1, SequenceEncoding.IS_SLIDER_BODY] = 1.0

    return grid
