from dataclasses import asdict, dataclass
from functools import partial
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from osu_fusion.library.osu.data.encode import BeatmapEncoding
from osu_fusion.library.osu.data.fit_bezier import fit_bezier, segment_length
from osu_fusion.library.osu.data.hit import decode_extents, decode_flips


@dataclass
class Metadata:
    audio_filename: str
    title: str
    artist: str
    version: str
    cs: float
    ar: float
    od: float
    hp: float


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
HPDrainRate: {hp}
CircleSize: {cs}
OverallDifficulty: {od}
ApproachRate: {ar}
SliderMultiplier: 1
SliderTickRate: 1

[TimingPoints]
{timing_points}

[HitObjects]
{hit_objects}
"""


def slider_decoder(
    cursor_signal: npt.NDArray,
    start_idx: int,
    end_idx: int,
    num_repeats: int,
) -> Tuple[float, List[npt.NDArray]]:
    first_slide_idx = round(start_idx + (end_idx - start_idx) / num_repeats)

    control_points = []
    length = 0.0

    path = fit_bezier(cursor_signal.T[start_idx : first_slide_idx + 1], max_err=50.0)
    for i, segment in enumerate(path):
        segment = segment.round()
        segment_length_ = segment_length(segment)
        if len(path) > 1 and i == 0 and segment_length_ < 20:
            continue
        control_points.extend(segment)
        length += segment_length_

    return length, control_points


def add_hit_circle(cursor_signals: npt.NDArray, onset_loc: int, t: float, combo_bit: int) -> str:
    x, y = cursor_signals[:, onset_loc].round().astype(int)
    return f"{x},{y},{t},{2**0 + combo_bit},0,0:0:0:0:"


def decode_beatmap(metadata: Metadata, encoded_beatmap: npt.NDArray, frame_times: npt.NDArray) -> str:  # noqa: C901
    cursor_signals = encoded_beatmap[[BeatmapEncoding.CURSOR_X, BeatmapEncoding.CURSOR_Y]]
    cursor_signals = ((cursor_signals + 1) / 2) * np.array([[512], [384]])

    hit_locs = decode_flips(encoded_beatmap[BeatmapEncoding.HIT])
    loc2idx = np.full_like(frame_times, -1, dtype=int)
    for i, onset_idx in enumerate(hit_locs):
        loc2idx[onset_idx] = i

    new_combos = [False] * len(hit_locs)
    for combo_locs in decode_flips(encoded_beatmap[BeatmapEncoding.COMBO]):
        new_combos[loc2idx[combo_locs]] = True

    sustain_ends = [-1] * len(hit_locs)
    for sustain_start, sustain_end in zip(*decode_extents(encoded_beatmap[BeatmapEncoding.SUSTAIN])):
        onset_idx = loc2idx[sustain_start]
        if onset_idx == -1:
            continue
        sustain_ends[onset_idx] = sustain_end

    slider_ends = [-1] * len(hit_locs)
    for slider_start, slider_end in zip(*decode_extents(encoded_beatmap[BeatmapEncoding.SLIDER])):
        onset_idx = loc2idx[slider_start]
        if onset_idx == -1:
            continue
        slider_ends[onset_idx] = slider_end

    timing_points = []
    hit_objects = []

    slider_ts = []
    slider_vels = []

    for hit_loc, new_combo, sustain_end, slider_end in zip(hit_locs, new_combos, sustain_ends, slider_ends):
        t = frame_times[hit_loc]
        combo_bit = 2**2 if new_combo else 0

        add_hit_circle_ = partial(add_hit_circle, cursor_signals, hit_loc, t, combo_bit)

        if sustain_end == -1:
            hit_objects.append(add_hit_circle_())
            continue

        u = frame_times[sustain_end]
        if u - t < 20:
            hit_objects.append(add_hit_circle_())
            continue

        if slider_end == -1:
            # Spinner
            hit_objects.append(f"256,192,{t},{2**3 + combo_bit},0,{u}")
            continue

        # Slider
        num_slides = max(1, round((sustain_end - hit_loc) / (slider_end - hit_loc)))
        length, control_points = slider_decoder(cursor_signals, hit_loc, sustain_end, num_slides)

        if length == 0:
            # zero-length slider
            hit_objects.append(add_hit_circle_())

        x1, y1 = control_points[0]
        curve_points = "|".join(f"{x}:{y}" for x, y in control_points[1:])
        hit_objects.append(f"{x1},{y1},{t},{2**1 + combo_bit},0,B|{curve_points},{num_slides},{length}")
        slider_ts.append(t)
        slider_vels.append(length * num_slides / (u - t))

    base_slider_vel = (min(slider_vels) * max(slider_vels)) ** 0.5
    beat_len = 100 / base_slider_vel

    # TODO: compute timing points using timing_signals
    timing_points.append(f"0,{beat_len},4,0,0,50,1,0")

    for t, vel in zip(slider_ts, slider_vels):
        slider_velocity = vel / base_slider_vel
        if slider_velocity > 10 or slider_velocity < 0.1:
            print(f"Warning: slider velocity {slider_velocity} is out of bounds, slider will not be good")
        timing_points.append(f"{t},{-100/slider_velocity},4,0,0,50,0,0")

    return map_template.format(
        **asdict(metadata),
        timing_points="\n".join(timing_points),
        hit_objects="\n".join(hit_objects),
    )
