from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import signal

from osu_fusion.data.enum import BeatmapEncoding
from osu_fusion.data.hit import decode_extents, decode_flips
from osu_fusion.osu.beatmap import TimingPoint

BEAT_DIVISOR = 8
SLIDER_MULT = 1.0
MIN_BPM = 1
MAX_BPM = 300


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


def get_timings(hit_times: npt.NDArray, timing_beat_len: float) -> Tuple[bool, TimingPoint]:
    offsets = hit_times % timing_beat_len
    hist, bin_edges = np.histogram(offsets, bins=100, range=(0, timing_beat_len))
    offset = bin_edges[np.argmax(hist)]
    return True, TimingPoint(offset, timing_beat_len, None, 4, False)


def calculate_timing_point(
    hit_times: npt.NDArray,
    allow_beat_snap: bool,
    verbose: bool = True,
) -> Tuple[bool, TimingPoint]:
    if not allow_beat_snap:
        return False, TimingPoint(0, 60000 / 200, None, 4, False)

    time_diffs = np.diff(hit_times)
    autocorr = signal.correlate(time_diffs, time_diffs, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    valid_periods = 60000 / np.arange(MIN_BPM, MAX_BPM + 1, 1)
    peaks, _ = signal.find_peaks(autocorr, distance=valid_periods.min())

    valid_peaks = peaks[(valid_periods.min() * 0.95 <= peaks) & (peaks <= valid_periods.max() * 1.05)]
    if len(valid_peaks) == 0:
        if verbose:
            print("Warning: no valid BPM found within the range, disabling beat snap")
        return False, TimingPoint(0, 60000 / 200, None, 4, False)

    best_peak = valid_peaks[np.argmax(autocorr[valid_peaks])]
    initial_bpm = 60000 / best_peak

    fine_tune_range = np.linspace(initial_bpm * 0.95, initial_bpm * 1.05, 1000)
    fine_tune_scores = np.zeros_like(fine_tune_range)
    for i, bpm in enumerate(fine_tune_range):
        beat_length = 60000 / bpm
        phase = hit_times % beat_length
        hist, _ = np.histogram(phase, bins=100, range=(0, beat_length))
        fine_tune_scores[i] = np.max(hist)

    best_bpm = fine_tune_range[np.argmax(fine_tune_scores)]
    return get_timings(hit_times, 60000 / best_bpm)


def snap_to_beat(t: float, u: float, beat_offset: float, beat_length: float) -> Tuple[float, float]:
    beat_f_len = beat_length / BEAT_DIVISOR
    t = round((t - beat_offset) / beat_f_len) * beat_f_len + beat_offset
    u = round((u - beat_offset) / beat_f_len) * beat_f_len + beat_offset
    return t, u


def decode_beatmap(  # noqa: C901
    metadata: Metadata,
    encoded_beatmap: npt.NDArray,
    frame_times: npt.NDArray,
    bpm: Optional[float],
    allow_beat_snap: bool = True,
    verbose: bool = True,
) -> str:
    hit_signals = encoded_beatmap[
        [
            BeatmapEncoding.HIT,
            BeatmapEncoding.SUSTAIN,
            BeatmapEncoding.SLIDER,
            BeatmapEncoding.WHITE_BEZIER_ANCHORS,
            BeatmapEncoding.RED_BEZIER_ANCHORS,
            BeatmapEncoding.LINE_ANCHORS,
            BeatmapEncoding.PERFECT_ANCHORS,
            BeatmapEncoding.COMBO,
        ]
    ]
    hit_signals = np.where(hit_signals > 0.0, 1.0, 0.0)  # Discretize signals
    cursor_signals = encoded_beatmap[[BeatmapEncoding.CURSOR_X, BeatmapEncoding.CURSOR_Y]]
    cursor_signals = ((cursor_signals + 1) / 2) * np.array([[512], [384]])

    hit_locs = decode_flips(hit_signals[BeatmapEncoding.HIT])
    loc2idx = np.full_like(frame_times, -1, dtype=int)
    for i, onset_idx in enumerate(hit_locs):
        loc2idx[onset_idx] = i

    new_combos = [False] * len(hit_locs)
    for combo_locs in decode_flips(hit_signals[BeatmapEncoding.COMBO]):
        new_combos[loc2idx[combo_locs]] = True

    sustain_ends = [-1] * len(hit_locs)
    for sustain_start, sustain_end in zip(*decode_extents(hit_signals[BeatmapEncoding.SUSTAIN]), strict=False):
        onset_idx = loc2idx[sustain_start]
        if onset_idx == -1:
            continue
        sustain_ends[onset_idx] = sustain_end

    slider_ends = [-1] * len(hit_locs)
    for slider_start, slider_end in zip(*decode_extents(hit_signals[BeatmapEncoding.SLIDER]), strict=False):
        onset_idx = loc2idx[slider_start]
        if onset_idx == -1:
            continue
        slider_ends[onset_idx] = slider_end

    white_bezier_anchor_locs = decode_flips(hit_signals[BeatmapEncoding.WHITE_BEZIER_ANCHORS])
    red_bezier_anchor_locs = decode_flips(hit_signals[BeatmapEncoding.RED_BEZIER_ANCHORS])
    linear_anchor_locs = decode_flips(hit_signals[BeatmapEncoding.LINE_ANCHORS])
    perfect_anchor_locs = decode_flips(hit_signals[BeatmapEncoding.PERFECT_ANCHORS])

    hos = []
    tps = []

    hit_times = frame_times[hit_locs]
    if bpm is not None:
        beat_snap, timing_point = get_timings(hit_times, 60000 / bpm)
    else:
        beat_snap, timing_point = calculate_timing_point(hit_times, allow_beat_snap, verbose)

    beat_length = timing_point.beat_length
    base_slider_vel = SLIDER_MULT * 100 / beat_length
    beat_offset = timing_point.t
    tps.append(f"{timing_point.t},{timing_point.beat_length},{timing_point.meter},0,0,50,1,0")

    for hit_loc, new_combo, sustain_end, slider_end in zip(
        hit_locs,
        new_combos,
        sustain_ends,
        slider_ends,
        strict=False,
    ):
        with np.errstate(invalid="raise"):
            x, y = cursor_signals[:, hit_loc].round().astype(int)
        t = frame_times[hit_loc]
        u = frame_times[sustain_end]
        combo_bit = 2**2 if new_combo else 0

        if beat_snap:
            t, u = snap_to_beat(t, u, beat_offset, beat_length)

        if sustain_end == -1:
            # No sustain
            hos.append(f"{x},{y},{t},{2**0 + combo_bit},0,0:0:0:0:")
            continue

        if sustain_end - hit_loc < 4:
            # Sustain too short
            hos.append(f"{x},{y},{t},{2**0 + combo_bit},0,0:0:0:0:")
            continue

        if slider_end == -1:
            # Spinner
            hos.append(f"256,192,{t},{2**3 + combo_bit},0,{u}")
            continue

        if slider_end - hit_loc < 4:
            # Slider too short
            hos.append(f"{x},{y},{t},{2**0 + combo_bit},0,0:0:0:0:")
            continue

        # Slider
        anchor_frames = []
        for frame in range(hit_loc + 1, slider_end):
            if frame in red_bezier_anchor_locs:
                anchor_frames.append((frame, "R"))
            elif frame in white_bezier_anchor_locs:
                anchor_frames.append((frame, "B"))
            elif frame in linear_anchor_locs:
                anchor_frames.append((frame, "L"))
            elif frame in perfect_anchor_locs:
                anchor_frames.append((frame, "P"))

        control_points = [(x, y)]
        with np.errstate(invalid="raise"):
            for frame_idx, anchor_type in anchor_frames:
                ax, ay = cursor_signals[:, frame_idx].round().astype(int)
                control_points.append((ax, ay))
                if anchor_type == "R":  # Red anchor, duplicate the point
                    control_points.append((ax, ay))
            end_x, end_y = cursor_signals[:, slider_end].round().astype(int)
            control_points.append((end_x, end_y))

        if any(anchor[1] in ("B", "R") for anchor in anchor_frames):
            slider_char = "B"
        elif anchor_frames:
            first_anchor_type = anchor_frames[0][1]
            slider_char = "L" if first_anchor_type == "L" else "P" if first_anchor_type == "P" else "B"
        else:
            slider_char = "B"

        length = 0.0
        for k in range(len(control_points) - 1):
            p1, p2 = np.array(control_points[k]), np.array(control_points[k + 1])
            length += np.linalg.norm(p2 - p1)

        if length < 1e-6:
            hos.append(f"{x},{y},{t},{2**0 + combo_bit},0,0:0:0:0:")
            continue

        num_slides = max(1, round((sustain_end - hit_loc) / (slider_end - hit_loc)))
        curve_points_str = "|".join(f"{px}:{py}" for px, py in control_points[1:])
        hos.append(f"{x},{y},{t},{2**1 + combo_bit},0,{slider_char}|{curve_points_str},{num_slides},{length:.2f}")

        vel = length * num_slides / (u - t)
        slider_vel = vel / base_slider_vel
        slider_vel = 1 if slider_vel == 0 else slider_vel
        if (slider_vel > 10 or slider_vel < 0.1) and verbose:
            print(f"Warning: slider velocity {slider_vel} is out of bounds, slider will not be good")
        tps.append(f"{t},{-100 / slider_vel},4,0,0,50,0,0")

    return map_template.format(
        **asdict(metadata),
        timing_points="\n".join(tps),
        hit_objects="\n".join(hos),
    )
