from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import signal
from slider.beatmap import Circle, Slider, Spinner, TimingPoint
from slider.curve import Curve, MultiBezier
from slider.position import Position

from osu_fusion.data.event import EventType
from osu_fusion.data.slider_path import SliderPath, position_to_progress

BEAT_DIVISOR = 8
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
    slider_multiplier: float
    slider_tick_rate: float


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
SliderMultiplier: {slider_multiplier}
SliderTickRate: {slider_tick_rate}

[TimingPoints]
{timing_points}

[HitObjects]
{hit_objects}
"""


def get_timings(hit_times: npt.NDArray, timing_beat_len: float) -> Tuple[bool, TimingPoint]:
    offsets = hit_times % timing_beat_len
    hist, bin_edges = np.histogram(offsets, bins=100, range=(0, timing_beat_len))
    offset = bin_edges[np.argmax(hist)]
    return True, TimingPoint(
        offset=timedelta(milliseconds=offset),
        ms_per_beat=timing_beat_len,
        meter=4,
        sample_type=0,
        sample_set=0,
        volume=50,
        parent=None,
        kiai_mode=False,
    )


def calculate_timing_point(
    hit_times: npt.NDArray,
    allow_beat_snap: bool,
    verbose: bool = True,
) -> Tuple[bool, TimingPoint]:
    if not allow_beat_snap:
        return False, TimingPoint(
            offset=timedelta(milliseconds=0),
            ms_per_beat=60000 / 200,
            meter=4,
            sample_type=0,
            sample_set=0,
            volume=50,
            parent=None,
            kiai_mode=False,
        )

    time_diffs = np.diff(hit_times)
    autocorr = signal.correlate(time_diffs, time_diffs, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    valid_periods = 60000 / np.arange(MIN_BPM, MAX_BPM + 1, 1)
    peaks, _ = signal.find_peaks(autocorr, distance=valid_periods.min())

    valid_peaks = peaks[(valid_periods.min() * 0.95 <= peaks) & (peaks <= valid_periods.max() * 1.05)]
    if len(valid_peaks) == 0:
        if verbose:
            print("Warning: no valid BPM found within the range, disabling beat snap")
        return False, TimingPoint(
            offset=timedelta(milliseconds=0),
            ms_per_beat=60000 / 200,
            meter=4,
            sample_type=0,
            sample_set=0,
            volume=50,
            parent=None,
            kiai_mode=False,
        )

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
    bpm: Optional[float],
    allow_beat_snap: bool = True,
    verbose: bool = True,
) -> str:
    seq = np.swapaxes(encoded_beatmap, 0, 1)
    seq_len = seq.shape[0]
    hit_objects = []
    timing_points = []
    curr_object = None
    curr_slider_path = []
    curr_slider_type = None
    span_duration = 0

    hit_times = seq[:, 2]
    if bpm is not None:
        beat_snap, timing_point = get_timings(hit_times, 60000 / bpm)
    else:
        beat_snap, timing_point = calculate_timing_point(hit_times, allow_beat_snap, verbose)

    timing_points.append(timing_point)

    for j in range(seq_len):
        x = round(float(seq[j, 0]))
        y = round(float(seq[j, 1]))
        time = timedelta(seconds=float(seq[j, 2] / 1000))
        type_index = int(np.argmax(seq[j, 3:]))
        pos = Position(x, y)

        if type_index == EventType.CIRCLE:
            hit_objects.append(Circle(pos, time, 0, new_combo=False))
        elif type_index == EventType.CIRCLE_NEW_COMBO:
            hit_objects.append(Circle(pos, time, 0, new_combo=True))
        elif type_index == EventType.SPINNER_START:
            curr_object = Spinner(pos, time, 0, time, new_combo=True)
        elif type_index == EventType.SPINNER_END and isinstance(curr_object, Spinner):
            curr_object.end_time = time
            hit_objects.append(curr_object)
        elif type_index == EventType.SLIDER_HEAD:
            curr_object = Slider(
                position=pos,
                time=time,
                combo_skip=0,
                end_time=time,
                hitsound=0,
                curve=MultiBezier([pos], 0),
                repeat=0,
                length=0,
                ticks=0,
                num_beats=0,
                tick_rate=metadata.slider_tick_rate,
                ms_per_beat=0,
                edge_sounds=[],
                edge_additions=[],
                new_combo=False,
            )
            curr_slider_path = [pos]
            curr_slider_type = "B"
        elif type_index == EventType.SLIDER_HEAD_NEW_COMBO:
            curr_object = Slider(
                position=pos,
                time=time,
                combo_skip=0,
                end_time=time,
                hitsound=0,
                curve=MultiBezier([pos], 0),
                repeat=0,
                length=0,
                ticks=0,
                num_beats=0,
                tick_rate=metadata.slider_tick_rate,
                ms_per_beat=0,
                edge_sounds=[],
                edge_additions=[],
                new_combo=True,
            )
            curr_slider_path = [pos]
            curr_slider_type = "B"
        elif type_index == EventType.BEZIER_ANCHOR and isinstance(curr_object, Slider):
            curr_slider_path.append(pos)
        elif type_index == EventType.PERFECT_ANCHOR and isinstance(curr_object, Slider):
            curr_slider_path.append(pos)
            curr_slider_type = "P"
        elif type_index == EventType.CATMULL_ANCHOR and isinstance(curr_object, Slider):
            curr_slider_path.append(pos)
            curr_slider_type = "C"
        elif type_index == EventType.LINEAR_ANCHOR and isinstance(curr_object, Slider):
            curr_slider_path.append(pos)
            curr_slider_path.append(pos)
        elif type_index == EventType.LAST_ANCHOR and isinstance(curr_object, Slider):
            curr_slider_path.append(pos)
            span_duration = (time - curr_object.time).total_seconds() * 1000
        elif type_index >= EventType.SLIDER_END_REPEAT_1 and isinstance(curr_object, Slider):
            slider_path = SliderPath(
                curr_slider_type,
                np.array(curr_slider_path, dtype=float),
            )
            req_length = slider_path.get_distance() * position_to_progress(
                slider_path,
                np.array(pos),
            )
            curr_object.curve = Curve.from_kind_and_points(
                curr_slider_type,
                [Position(p[0], p[1]) for p in slider_path.control_points],
                req_length,
            )
            curr_object.length = req_length
            curr_object.end_time = time
            duration = (time - curr_object.time).total_seconds() * 1000
            curr_object.repeat = (
                round(duration / span_duration)
                if type_index > EventType.SLIDER_END_REPEAT_3
                else type_index - EventType.SLIDER_END_REPEAT_1 + 1
            )
            curr_object.edge_sounds = [0] * curr_object.repeat
            curr_object.edge_additions = ["0:0"] * curr_object.repeat
            hit_objects.append(curr_object)

            tp = timing_point
            parent = tp.parent if tp.parent is not None else tp
            ms_per_beat = tp.parent.ms_per_beat if tp.parent is not None else tp.ms_per_beat
            global_sv = metadata.slider_multiplier
            new_sv_multiplier = req_length * ms_per_beat / (100 * global_sv * span_duration)
            timing_points.append(
                TimingPoint(
                    curr_object.time,
                    -100 / new_sv_multiplier if new_sv_multiplier > 0 else -100,
                    tp.meter,
                    tp.sample_type,
                    tp.sample_set,
                    tp.volume,
                    parent,
                    tp.kiai_mode,
                ),
            )

    return map_template.format(
        **asdict(metadata),
        timing_points="\n".join([tp.pack() for tp in timing_points]),
        hit_objects="\n".join([ho.pack() for ho in hit_objects]),
    )
