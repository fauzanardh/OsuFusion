from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy import signal
from slider.beatmap import TimingPoint

from osu_fusion.data.const import MS_PER_FRAME
from osu_fusion.data.encode import SequenceEncoding, LOG_SCALE_LENGTH, LOG_SCALE_REPEATS, TYPE_START, TYPE_END

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
    verbose: bool = True,
) -> Tuple[bool, TimingPoint]:
    if len(hit_times) < 2:
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
            print("Warning: no valid BPM found within the range")
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


def decode_kiai_sections(encoded_seq: npt.NDArray) -> List[Tuple[float, float]]:
    kiai_signal = encoded_seq[:, SequenceEncoding.IS_KIAI] > 0.0
    sections = []
    in_kiai = False
    start_ms = 0.0

    for i in range(len(kiai_signal)):
        if kiai_signal[i] and not in_kiai:
            start_ms = i * MS_PER_FRAME
            in_kiai = True
        elif not kiai_signal[i] and in_kiai:
            end_ms = i * MS_PER_FRAME
            sections.append((start_ms, end_ms))
            in_kiai = False

    if in_kiai:
        sections.append((start_ms, len(kiai_signal) * MS_PER_FRAME))

    return sections


def decode_sequence(  # noqa: C901
    metadata: Metadata,
    encoded_seq: npt.NDArray,
    verbose: bool = True,
) -> str:
    if encoded_seq.shape[0] == 0:
        return map_template.format(**asdict(metadata), timing_points="", hit_objects="")

    is_note_signal = encoded_seq[:, SequenceEncoding.IS_NOTE]
    peaks = np.where(is_note_signal > 0.0)[0]

    if len(peaks) == 0:
        return map_template.format(**asdict(metadata), timing_points="", hit_objects="")

    N_events = len(peaks)

    offsets_norm = (encoded_seq[peaks, SequenceEncoding.OFFSET] + 1.0) / 2.0
    offsets = np.clip(offsets_norm, 0.0, 1.0)

    absolute_times = (peaks + offsets) * MS_PER_FRAME

    xs = (encoded_seq[peaks, SequenceEncoding.X] + 1.0) * 256.0
    ys = (encoded_seq[peaks, SequenceEncoding.Y] + 1.0) * 192.0

    new_combos = encoded_seq[peaks, SequenceEncoding.NEW_COMBO] > 0.0

    norm_lengths = (encoded_seq[peaks, SequenceEncoding.SLIDER_LENGTH] + 1.0) / 2.0
    lengths = np.maximum(0, np.expm1(norm_lengths * LOG_SCALE_LENGTH))

    norm_repeats = (encoded_seq[peaks, SequenceEncoding.SLIDER_REPEATS] + 1.0) / 2.0
    repeats = np.maximum(1, np.round(np.expm1(norm_repeats * LOG_SCALE_REPEATS)))

    type_logits = encoded_seq[peaks, TYPE_START:TYPE_END]
    event_types = np.argmax(type_logits, axis=1) + TYPE_START

    tps: List[str] = []
    obj_times = []

    for i in range(N_events):
        if event_types[i] in (
            SequenceEncoding.TYPE_CIRCLE,
            SequenceEncoding.TYPE_SLIDER_HEAD,
            SequenceEncoding.TYPE_SPINNER,
        ):
            obj_times.append(absolute_times[i])

    if len(obj_times) > 0:
        _, timing_point = calculate_timing_point(np.array(obj_times), verbose)
        tps.append(
            f"{timing_point.offset.total_seconds() * 1000},{timing_point.ms_per_beat},{timing_point.meter},0,0,50,1,0",
        )
        beat_length = timing_point.ms_per_beat
    else:
        beat_length = 60000 / 200

    kiai_sections = decode_kiai_sections(encoded_seq)
    hos: List[str] = []

    in_slider = False
    in_spinner = False

    base_slider_vel = metadata.slider_multiplier * 100 / beat_length
    slider_x, slider_y, slider_time, slider_nc = 0, 0, 0, False
    slider_points = []
    slider_curve_type = "B"
    slider_length = 0.0
    slider_repeat = 1

    spinner_time, spinner_nc = 0, False

    def emit_slider(end_time: float) -> None:
        nonlocal in_slider
        if not in_slider:
            return

        c_type = slider_curve_type
        if len(slider_points) == 0:
            combo_bit = 2**2 if slider_nc else 0
            hos.append(f"{slider_x},{slider_y},{int(slider_time)},{2**0 + combo_bit},0,0:0:0:0:")
        else:
            if c_type == "B" and len(slider_points) == 1:
                c_type = "L"

            points_str = "|".join([f"{round(px)}:{round(py)}" for px, py in slider_points])
            combo_bit = 2**2 if slider_nc else 0

            sl_length = slider_length
            if sl_length < 1.0:
                sl_length = 10.0

            hos.append(
                f"{slider_x},{slider_y},{int(slider_time)},{2**1 + combo_bit},0,"
                f"{c_type}|{points_str},{int(slider_repeat)},{sl_length:.2f}",
            )

            duration_ms = end_time - slider_time
            if duration_ms > 0 and base_slider_vel > 0:
                vel = sl_length * slider_repeat / duration_ms
                slider_vel = vel / base_slider_vel
                slider_vel = max(0.1, min(10.0, slider_vel)) if slider_vel != 0 else 1.0
                tps.append(f"{int(slider_time)},{-100 / slider_vel},4,0,0,50,0,0")

        in_slider = False

    def emit_spinner(end_time: float) -> None:
        nonlocal in_spinner
        if not in_spinner:
            return
        combo_bit = 2**2 if spinner_nc else 0
        end_time = max(spinner_time + 1, end_time)
        hos.append(f"256,192,{int(spinner_time)},{2**3 + combo_bit},0,{int(end_time)}")
        in_spinner = False

    for i in range(N_events):
        t = absolute_times[i]
        x = round(xs[i])
        y = round(ys[i])
        nc = new_combos[i]
        evt = event_types[i]

        if evt in (SequenceEncoding.TYPE_CIRCLE, SequenceEncoding.TYPE_SLIDER_HEAD, SequenceEncoding.TYPE_SPINNER):
            if in_slider:
                emit_slider(t)
            if in_spinner:
                emit_spinner(t)

        if evt == SequenceEncoding.TYPE_CIRCLE:
            combo_bit = 2**2 if nc else 0
            hos.append(f"{x},{y},{int(t)},{2**0 + combo_bit},0,0:0:0:0:")

        elif evt == SequenceEncoding.TYPE_SPINNER:
            in_spinner = True
            spinner_time = t
            spinner_nc = nc

        elif evt == SequenceEncoding.TYPE_SPINNER_END:
            emit_spinner(t)

        elif evt == SequenceEncoding.TYPE_SLIDER_HEAD:
            in_slider = True
            slider_x, slider_y = x, y
            slider_time = t
            slider_nc = nc
            slider_points = []
            slider_curve_type = "B"
            slider_length = lengths[i]
            slider_repeat = repeats[i]

        elif evt in (
            SequenceEncoding.TYPE_BEZIER_ANCHOR,
            SequenceEncoding.TYPE_PERFECT_ANCHOR,
            SequenceEncoding.TYPE_CATMULL_ANCHOR,
            SequenceEncoding.TYPE_RED_ANCHOR,
            SequenceEncoding.TYPE_LAST_ANCHOR,
        ):
            if in_slider:
                slider_points.append((x, y))
                if evt == SequenceEncoding.TYPE_PERFECT_ANCHOR:
                    slider_curve_type = "P"
                elif evt == SequenceEncoding.TYPE_CATMULL_ANCHOR:
                    slider_curve_type = "C"

        elif evt == SequenceEncoding.TYPE_SLIDER_END and in_slider:
            emit_slider(t)

    if in_slider:
        emit_slider(absolute_times[-1] + 100)
    if in_spinner:
        emit_spinner(absolute_times[-1] + 100)

    tps.sort(key=lambda tp_str: float(tp_str.split(",")[0]))
    if kiai_sections:
        final_tps = []
        for tp_str in tps:
            fields = tp_str.split(",")
            tp_time = float(fields[0])
            in_kiai = any(start <= tp_time < end for start, end in kiai_sections)
            if in_kiai:
                fields[7] = str(int(fields[7]) | 1)
            else:
                fields[7] = str(int(fields[7]) & ~1)
            final_tps.append(",".join(fields))

        for start_ms, end_ms in kiai_sections:
            start_exists = any(abs(float(tp.split(",")[0]) - start_ms) < 1 for tp in final_tps)
            end_exists = any(abs(float(tp.split(",")[0]) - end_ms) < 1 for tp in final_tps)
            if not start_exists:
                final_tps.append(f"{int(start_ms)},-100,4,0,0,50,0,1")
            if not end_exists:
                final_tps.append(f"{int(end_ms)},-100,4,0,0,50,0,0")

        final_tps.sort(key=lambda tp_str: float(tp_str.split(",")[0]))
        tps = final_tps

    return map_template.format(
        **asdict(metadata),
        timing_points="\n".join(tps),
        hit_objects="\n".join(hos),
    )
