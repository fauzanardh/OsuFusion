import numpy as np
import numpy.typing as npt
from slider.beatmap import Beatmap, Slider, Spinner, Circle
from slider.curve import Perfect, Catmull
from enum import IntEnum
from typing import Any


class SequenceEncoding(IntEnum):
    TIME = 0
    X = 1
    Y = 2
    NEW_COMBO = 3
    SLIDER_LENGTH = 4
    SLIDER_REPEATS = 5

    TYPE_CIRCLE = 6
    TYPE_SLIDER_HEAD = 7
    TYPE_BEZIER_ANCHOR = 8
    TYPE_PERFECT_ANCHOR = 9
    TYPE_CATMULL_ANCHOR = 10
    TYPE_RED_ANCHOR = 11
    TYPE_LAST_ANCHOR = 12
    TYPE_SLIDER_END = 13
    TYPE_SPINNER = 14
    TYPE_SPINNER_END = 15
    TYPE_WAIT = 16
    TYPE_PAD = 17


SEQ_DIM = len(SequenceEncoding)

MAX_DELTA_TIME = 1000.0
LOG_SCALE_LENGTH = np.log1p(1000.0)
LOG_SCALE_REPEATS = np.log1p(10.0)


def get_anchor_type(curve: Any, prev_point: Any, point: Any) -> int:  # noqa: ANN401
    if prev_point is not None and (point.x, point.y) == (prev_point.x, prev_point.y):
        return SequenceEncoding.TYPE_RED_ANCHOR

    if isinstance(curve, Perfect):
        return SequenceEncoding.TYPE_PERFECT_ANCHOR
    elif isinstance(curve, Catmull):
        return SequenceEncoding.TYPE_CATMULL_ANCHOR
    else:
        return SequenceEncoding.TYPE_BEZIER_ANCHOR


def encode_sequence(beatmap: Beatmap) -> npt.NDArray:  # noqa: C901
    events = []
    prev_time = 0.0

    def add_event(
        time: float,
        x: float,
        y: float,
        new_combo: bool,
        length: float,
        repeats: int,
        event_type: int,
    ) -> None:
        nonlocal prev_time
        delta_time = time - prev_time

        while delta_time > MAX_DELTA_TIME:
            vec_w = np.full(SEQ_DIM, -1.0, dtype=np.float32)
            vec_w[SequenceEncoding.TIME] = 1.0
            vec_w[SequenceEncoding.X] = 0.0
            vec_w[SequenceEncoding.Y] = 0.0
            vec_w[SequenceEncoding.TYPE_WAIT] = 1.0
            events.append(vec_w)
            prev_time += MAX_DELTA_TIME
            delta_time -= MAX_DELTA_TIME

        prev_time = time
        vec = np.full(SEQ_DIM, -1.0, dtype=np.float32)

        vec[SequenceEncoding.TIME] = (max(0.0, delta_time) / (MAX_DELTA_TIME / 2)) - 1.0
        vec[SequenceEncoding.X] = (x / 256.0) - 1.0 if x is not None else 0.0
        vec[SequenceEncoding.Y] = (y / 192.0) - 1.0 if y is not None else 0.0
        vec[SequenceEncoding.NEW_COMBO] = 1.0 if new_combo else -1.0
        vec[SequenceEncoding.SLIDER_LENGTH] = (np.log1p(max(0.0, length)) / LOG_SCALE_LENGTH) * 2.0 - 1.0
        vec[SequenceEncoding.SLIDER_REPEATS] = (np.log1p(max(0.0, repeats)) / LOG_SCALE_REPEATS) * 2.0 - 1.0
        vec[event_type] = 1.0

        events.append(vec)

    for ho in beatmap.hit_objects():
        time_ms = ho.time.total_seconds() * 1000.0

        if isinstance(ho, Circle):
            add_event(time_ms, ho.position.x, ho.position.y, ho.new_combo, 0, 0, SequenceEncoding.TYPE_CIRCLE)

        elif isinstance(ho, Spinner):
            add_event(time_ms, 256, 192, ho.new_combo, 0, 0, SequenceEncoding.TYPE_SPINNER)
            end_time_ms = ho.end_time.total_seconds() * 1000.0
            add_event(end_time_ms, 256, 192, 0, 0, 0, SequenceEncoding.TYPE_SPINNER_END)

        elif isinstance(ho, Slider):
            head_time = time_ms
            add_event(
                head_time,
                ho.position.x,
                ho.position.y,
                ho.new_combo,
                ho.length,
                ho.repeat,
                SequenceEncoding.TYPE_SLIDER_HEAD,
            )

            points = ho.curve.points
            if len(points) > 2:
                for i in range(1, len(points) - 1):
                    anchor_type = get_anchor_type(ho.curve, points[i - 1], points[i])
                    add_event(head_time, points[i].x, points[i].y, False, 0, 0, anchor_type)

            if len(points) > 1:
                add_event(head_time, points[-1].x, points[-1].y, False, 0, 0, SequenceEncoding.TYPE_LAST_ANCHOR)

            end_time_ms = ho.end_time.total_seconds() * 1000.0
            end_pos = ho.curve(1.0) if ho.repeat % 2 != 0 else ho.position
            add_event(end_time_ms, end_pos.x, end_pos.y, False, ho.length, ho.repeat, SequenceEncoding.TYPE_SLIDER_END)

    if len(events) == 0:
        return np.zeros((0, SEQ_DIM), dtype=np.float32)

    return np.stack(events, axis=0)
