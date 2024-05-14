from typing import List

import numpy as np
import numpy.typing as npt

from osu_fusion.library.osu.beatmap import Beatmap, TimingPoint


def timing_signal(beatmap: Beatmap, frame_times: npt.NDArray) -> npt.NDArray:
    signals = np.zeros((2, frame_times.shape[0]))
    for i, timing_point in enumerate(beatmap.uninherited_timing_points):
        start = timing_point.t
        if i == 0:
            measure_length = timing_point.beat_length * timing_point.meter
            start -= (start // measure_length + 1) * measure_length
        window = frame_times >= start
        beat_phase = (frame_times - start) / timing_point.beat_length
        measure_phase = beat_phase / timing_point.meter
        signals[0, window] = beat_phase[window] % 1
        signals[1, window] = measure_phase[window] % 1

    return signals


def decode_timing_signal(timing_signal: npt.NDArray, frame_times: npt.NDArray) -> List[TimingPoint]:
    timing_points = []
    for i in range(timing_signal.shape[1]):
        beat_phase = timing_signal[0, i]
        measure_phase = timing_signal[1, i]
        t = frame_times[i]
        beat_length = 1 / beat_phase if beat_phase != 0 else 1  # Should've been float('inf')
        meter = measure_phase if measure_phase != 0 else 1
        timing_points.append(TimingPoint(t, beat_length, None, meter, None))
    return timing_points
