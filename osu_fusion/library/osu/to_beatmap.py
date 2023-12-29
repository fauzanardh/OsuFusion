import bisect
from typing import Dict, List, Optional, Tuple, Union

import bezier
import numpy as np
import numpy.typing as npt
import scipy

from osu_fusion.library.osu.from_beatmap import CURSOR_DIM, HIT_DIM, SLIDER_DIM
from osu_fusion.library.osu.hit_objects import TimingPoint
from osu_fusion.library.osu.utils.fit_bezier import fit_bezier
from osu_fusion.library.osu.utils.smooth_hit import decode_hit, decode_hold

BEAT_DIVISOR = 4

map_template = """osu file format v14

[General]
AudioFilename: {audio_filename}
AudioLeadIn: 0
Mode: 0

[Metadata]
Title: {title}
TitleUnicode: {title}
Artist: {artist}
ArtistUnicode: {artist}
Creator: OsuFusion
Version: {version}
Tags: OsuFusion

[Difficulty]
HPDrainRate: 5
CircleSize: 4
OverallDifficulty: 9.5
ApproachRate: 9.5
SliderMultiplier: 1
SliderTickRate: 1

[TimingPoints]
{timing_points}

[HitObjects]
{hit_objects}
"""


def to_sorted_hits(signals: npt.NDArray) -> List[Tuple[npt.NDArray, npt.NDArray, int, bool]]:
    tap_signal, slider_hold_signal, spinner_signal, new_combo_signal = signals

    tap_idxs = decode_hit(tap_signal)
    slider_hold_start_idxs, slider_hold_end_idxs = decode_hold(slider_hold_signal)
    spinner_start_idxs, spinner_end_idxs = decode_hold(spinner_signal)
    new_combo_idxs = decode_hit(new_combo_signal)

    sorted_hits = sorted(
        [
            *[(t, t, 0, False) for t in tap_idxs],
            *[(s, e, 1, False) for s, e in zip(sorted(slider_hold_start_idxs), sorted(slider_hold_end_idxs))],
            *[(s, e, 2, False) for s, e in zip(sorted(spinner_start_idxs), sorted(spinner_end_idxs))],
        ],
    )

    for new_combo_idx in new_combo_idxs:
        idx = bisect.bisect_left(sorted_hits, (new_combo_idx,))
        if (idx == len(sorted_hits)) or (
            idx > 0 and abs(new_combo_idx - sorted_hits[idx][0]) > abs(sorted_hits[idx - 1][0] - new_combo_idx)
        ):
            idx -= 1
        sorted_hits[idx] = (*sorted_hits[idx][:3], True)

    return sorted_hits


def to_playfield_coordinates(cursor_signal: npt.NDArray) -> npt.NDArray:
    return (cursor_signal + 1) / 2 * np.array([[512], [384]])


def to_slider_decoder(processed_cursor_signals: npt.NDArray, slider_signal: npt.NDArray) -> callable:
    repeat_signal = slider_signal[0]
    repeat_idxs = decode_hit(repeat_signal)

    def decoder(a: int, b: int) -> Tuple[float, int, npt.NDArray]:
        repeat_idx_in_range = [r for r in repeat_idxs if a < r < b]
        if len(repeat_idx_in_range):
            r = repeat_idx_in_range[0]
            slides = round((b - a) / (r - a))
        else:
            slides = 1

        r = round(a + (b - a) / slides)
        length = 0
        control_points = []
        slider_segments = processed_cursor_signals.T[a : r + 1]
        for _bezier in fit_bezier(slider_segments, max_err=100):
            _bezier_np = np.array(_bezier).round().astype(int)
            control_points.extend(_bezier_np)
            length += bezier.Curve.from_nodes(_bezier_np.T).length
        return length, slides, control_points

    return decoder


def to_beatmap(  # noqa: C901
    metadata: Dict,
    signals: npt.NDArray,
    frame_times: npt.NDArray,
    timing: Optional[Union[int, List[TimingPoint]]],
) -> str:
    hit_signal, signals = np.split(signals, (HIT_DIM,))
    slider_signal, signals = np.split(signals, (SLIDER_DIM,))
    cursor_signal, signals = np.split(signals, (CURSOR_DIM,))
    assert signals.shape[0] == 0

    sorted_hits = to_sorted_hits(hit_signal)
    processed_cursor_signals = to_playfield_coordinates(cursor_signal)
    slider_decoder = to_slider_decoder(processed_cursor_signals, slider_signal)

    if isinstance(timing, list) and len(timing) > 0:
        beat_snap, timing_points = True, timing
    elif timing is None:
        # TODO: Compute beatmap timing from hit times
        beat_snap, timing_points = False, [TimingPoint(0, 60000 / 200, None, 4, None)]
    elif isinstance(timing, (int, float)):
        timing_beat_length = 60.0 * 1000.0 / float(timing)
        offset_dist = scipy.stats.gaussian_kde([frame_times[i] % timing_beat_length for i, _, _, _ in sorted_hits])
        offset = offset_dist.pdf(np.linspace(0, timing_beat_length, 1000)).argmax() / 1000.0 * timing_beat_length
        beat_snap, timing_points = True, [TimingPoint(offset, timing_beat_length, None, 4, None)]

    hit_objects_str = []
    timing_points_str = []

    beat_length = timing_points[0].beat_length
    base_slider_velocity = 100 / beat_length
    beat_offset = timing_points[0].t

    def add_circle(i: int, j: int, t: int, u: int, new_combo: bool) -> None:
        x, y = processed_cursor_signals[:, i].round().astype(int)
        hit_objects_str.append(f"{x},{y},{t},{1 + new_combo},0,0:0:0:0:")

    def add_spinner(i: int, j: int, t: int, u: int, new_combo: bool) -> None:
        if t == u:
            return add_circle(i, j, t, u, new_combo)
        hit_objects_str.append(f"256,192,{t},{8 + new_combo},0,{u}")

    def add_slider(i: int, j: int, t: int, u: int, new_combo: bool) -> None:
        if t == u:
            return add_circle(i, j, t, u, new_combo)

        length, slides, control_points = slider_decoder(i, j)

        if length == 0:
            return add_circle(i, j, t, u, new_combo)

        slider_velocity = length * slides / (u - t) / base_slider_velocity
        if slider_velocity > 10 or slider_velocity < 0.1:
            print(f"Warning: slider velocity {slider_velocity} is out of bounds, slider will not be good")

        x1, y1 = control_points[0]
        curve_points = "|".join(f"{x}:{y}" for x, y in control_points[1:])
        hit_objects_str.append(f"{x1},{y1},{t},{2 + new_combo},0,B|{curve_points},{slides},{length}")

        if len(timing_points_str) == 0:
            print("Warning: inherited timing point added before any uninherited timing points")
        timing_points_str.append(f"{t},{-100/slider_velocity},4,0,0,50,0,0")

    last_up = None
    for i, j, object_type, new_combo in sorted_hits:
        t, u = frame_times[i], frame_times[j]
        if beat_snap:
            beat_length_frac = beat_length / BEAT_DIVISOR
            t = round((t - beat_offset) / beat_length_frac) * beat_length_frac + beat_offset
            u = round((u - beat_offset) / beat_length_frac) * beat_length_frac + beat_offset

        t, u = int(t), int(u)

        if len(timing_points) > 0 and t > timing_points[0].t:
            timing_point = timing_points.pop(0)
            timing_points_str.append(f"{timing_point.t},{timing_point.beat_length},{timing_point.meter},0,0,50,1,0")
            beat_length = timing_point.beat_length
            base_slider_velocity = 100 / beat_length
            beat_offset = timing_point.t

        if last_up is not None and t <= last_up + 1:
            continue
        [add_circle, add_slider, add_spinner][object_type](i, j, t, u, 4 if new_combo else 0)
        last_up = u

    return map_template.format(
        **metadata,
        timing_points="\n".join(timing_points_str),
        hit_objects="\n".join(hit_objects_str),
    )
