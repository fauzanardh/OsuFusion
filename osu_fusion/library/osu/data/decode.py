from dataclasses import asdict, dataclass
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import signal

from osu_fusion.library.osu.beatmap import TimingPoint
from osu_fusion.library.osu.data.encode import BeatmapEncoding
from osu_fusion.library.osu.data.fit_bezier import fit_bezier, segment_length
from osu_fusion.library.osu.data.hit import decode_extents, decode_flips

BEAT_DIVISOR = 16
SLIDER_MULT = 1.0
MIN_BPM = 1
MAX_BPM = 300


@dataclass
class Metadata:
    audio_filename: str
    title: str
    artist: str
    bpm: Optional[float]
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


def get_timings(hit_times: npt.NDArray, timing_beat_len: float) -> Tuple[bool, TimingPoint]:
    offsets = hit_times % timing_beat_len
    hist, bin_edges = np.histogram(offsets, bins=100, range=(0, timing_beat_len))
    offset = bin_edges[np.argmax(hist)]
    return True, TimingPoint(offset, timing_beat_len, None, 4, None)


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

    hos = []
    tps = []

    if metadata.bpm is not None:
        timing_beat_len = 60000 / metadata.bpm
        hit_times = frame_times[hit_locs]
        beat_snap, timing_point = get_timings(hit_times, timing_beat_len)
    else:
        hit_times = frame_times[hit_locs]
        time_diffs = np.diff(hit_times)
        autocorr = signal.correlate(time_diffs, time_diffs, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        valid_periods = 60000 / np.arange(MIN_BPM, MAX_BPM + 1)
        peaks, _ = signal.find_peaks(autocorr, distance=valid_periods.min())

        valid_peaks = peaks[(valid_periods.min() <= peaks) & (peaks <= valid_periods.max())]
        if len(valid_peaks) == 0:
            print("Warning: no valid BPM found within the range, disabling beat snap")
            beat_snap, timing_point = False, TimingPoint(0, 60000 / 200, None, 4, None)
        else:
            best_period = valid_peaks[np.argmax(autocorr[valid_peaks])]
            bpm = 60000 / best_period
            timing_beat_len = 60000 / bpm
            beat_snap, timing_point = get_timings(hit_times, timing_beat_len)

    beat_length = timing_point.beat_length
    base_slider_vel = SLIDER_MULT * 100 / beat_length
    beat_offset = timing_point.t
    tps.append(f"{timing_point.t},{timing_point.beat_length},{timing_point.meter},0,0,50,1,0")

    last_up = None
    for hit_loc, new_combo, sustain_end, slider_end in zip(hit_locs, new_combos, sustain_ends, slider_ends):
        t = frame_times[hit_loc]
        u = frame_times[sustain_end]
        combo_bit = 2**2 if new_combo else 0

        if beat_snap:
            beat_f_len = beat_length / BEAT_DIVISOR
            t = round((t - beat_offset) / beat_f_len) * beat_f_len + beat_offset
            u = round((u - beat_offset) / beat_f_len) * beat_f_len + beat_offset

        if last_up is not None and t <= last_up + 1:
            continue

        _add_hit_circle = partial(add_hit_circle, cursor_signals, hit_loc, t, combo_bit)

        if sustain_end == -1:
            hos.append(_add_hit_circle())
            continue

        if u - t < 20:
            hos.append(_add_hit_circle())
            continue

        if slider_end == -1:
            # Spinner
            hos.append(f"256,192,{t},{2**3 + combo_bit},0,{u}")
            continue

        # Slider
        num_slides = max(1, round((sustain_end - hit_loc) / (slider_end - hit_loc)))
        length, control_points = slider_decoder(cursor_signals, hit_loc, sustain_end, num_slides)

        if length == 0:
            # zero-length slider
            hos.append(_add_hit_circle())

        x1, y1 = control_points[0]
        curve_points = "|".join(f"{x}:{y}" for x, y in control_points[1:])
        hos.append(f"{x1},{y1},{t},{2**1 + combo_bit},0,B|{curve_points},{num_slides},{length}")

        if len(tps) == 0:
            print("Warning: inherited timing point added before any uninherited timing points")
        vel = length * num_slides / (u - t)
        slider_vel = vel / base_slider_vel
        if slider_vel > 10 or slider_vel < 0.1:
            print(f"Warning: slider velocity {slider_vel} is out of bounds, slider will not be good")
        tps.append(f"{t},{-100/slider_vel},4,0,0,50,0,0")

        last_up = u

    return map_template.format(
        **asdict(metadata),
        timing_points="\n".join(tps),
        hit_objects="\n".join(hos),
    )
