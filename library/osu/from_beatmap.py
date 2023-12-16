from typing import Generator, Tuple

import numpy as np
import numpy.typing as npt

from library.osu.beatmap import Beatmap
from library.osu.hit_objects import Circle, HitObject, Slider, Spinner
from library.osu.utils.smooth_hit import encode_hit, encode_hold

HIT_DIM = 4
AUDIO_DIM = 40
SLIDER_DIM = 1
CURSOR_DIM = 2
TOTAL_DIM = HIT_DIM + SLIDER_DIM + CURSOR_DIM
CONTEXT_DIM = 4


def hit_object_pairs(
    beatmap: Beatmap,
    frame_times: npt.ArrayLike,
) -> Generator[Tuple[HitObject, HitObject], None, None]:
    pairs = zip([None, *beatmap.hit_objects], [*beatmap.hit_objects, None])
    _curr, _next = next(pairs)
    for t in frame_times:
        while _next is not None and _next.t <= t:
            _curr, _next = _next, next(pairs)
        yield _curr, _next


def hit_signal(beatmap: Beatmap, frame_times: npt.ArrayLike) -> npt.ArrayLike:
    hit_signals = np.full((4, frame_times.shape[0]), -1.0)
    for hit_object in beatmap.hit_objects:
        if isinstance(hit_object, Circle):
            encode_hit(hit_signals[0], frame_times, hit_object.t)
        elif isinstance(hit_object, Slider):
            encode_hold(hit_signals[1], frame_times, hit_object.t, hit_object.end_time())
        elif isinstance(hit_object, Spinner):
            encode_hold(hit_signals[2], frame_times, hit_object.t, hit_object.end_time())

        if hit_object.new_combo:
            encode_hit(hit_signals[3], frame_times, hit_object.t)

    return hit_signals


def slider_signal(beatmap: Beatmap, frame_times: npt.ArrayLike) -> npt.ArrayLike:
    slider_signals = np.full((2, frame_times.shape[0]), -1.0)

    for hit_object in beatmap.hit_objects:
        if not isinstance(hit_object, Slider):
            continue
        if hit_object.slides > 1:
            encode_hit(slider_signals[0], frame_times, hit_object.t + hit_object.slider_duration / hit_object.slides)

    return slider_signals


def cursor_signal(beatmap: Beatmap, frame_times: npt.ArrayLike) -> npt.ArrayLike:
    pos = []
    for t, (_curr, _next) in zip(frame_times, hit_object_pairs(beatmap, frame_times)):
        if _curr is None:
            pos.append(_next.start_pos())
        elif t < _curr.end_time():
            if isinstance(_curr, Spinner):
                pos.append(_curr.start_pos())
            else:
                single_slide = _curr.slide_duration / _curr.slides
                ts = (t - _curr.t) % (2 * single_slide) / single_slide
                if ts < 1:
                    pos.append(_curr.lerp(ts))
                else:
                    pos.append(_curr.lerp(2 - ts))
        elif _next is None:
            pos.append(_curr.end_pos())
        else:
            f = (t - _curr.end_time()) / (_next.t - _curr.end_time())
            pos.append((1 - f) * _curr.end_pos() + f * _next.start_pos())

    cursor_signals = np.array(pos).T / np.array([512, 384])
    cursor_signals = cursor_signals * 2 - 1

    return cursor_signals


def from_beatmap(beatmap: Beatmap, frame_times: npt.ArrayLike) -> npt.ArrayLike:
    signals = np.concatenate(
        [
            hit_signal(beatmap, frame_times),
            slider_signal(beatmap, frame_times),
            cursor_signal(beatmap, frame_times),
        ],
    )
    assert signals.shape[0] == TOTAL_DIM, f"Expected {TOTAL_DIM} dimensions, got {signals.shape[0]}"

    return signals
