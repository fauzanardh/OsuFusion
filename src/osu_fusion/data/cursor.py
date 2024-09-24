import numpy as np
import numpy.typing as npt

from osu_fusion.osu.beatmap import Beatmap
from osu_fusion.osu.hit_objects import Circle, Slider, Spinner


def cursor_signal(beatmap: Beatmap, frame_times: npt.NDArray) -> npt.NDArray:
    preempt = 1200 + (120 if beatmap.ar <= 5 else 150) * (5 - beatmap.ar)

    start = Circle(0, True, 256, 192)
    hit_objects = [start, *beatmap.hit_objects]
    positions = []

    for current_obj, next_obj in zip(hit_objects, hit_objects[1:] + [None], strict=True):
        if isinstance(current_obj, Spinner):
            current_count = np.sum((frame_times >= current_obj.t) & (frame_times < current_obj.end_time()))
            positions.extend(current_obj.start_pos()[None].repeat(current_count, axis=0))
        elif isinstance(current_obj, Slider):
            current_t = frame_times[(frame_times >= current_obj.t) & (frame_times < current_obj.end_time())]
            current_f = (current_t - current_obj.t) % (current_obj.slide_duration * 2) / current_obj.slide_duration
            if len(current_f) > 0:
                positions.extend(current_obj.lerp(np.where(current_f < 1, current_f, 2 - current_f)))

        if next_obj is None:
            map_end_count = np.sum(frame_times >= current_obj.end_time())
            positions.extend(current_obj.end_pos()[None].repeat(map_end_count, axis=0))
            break

        wait_count = np.sum((frame_times >= current_obj.end_time()) & (frame_times < next_obj.t - preempt))
        positions.extend(current_obj.end_pos()[None].repeat(wait_count, axis=0))

        start_time = max(current_obj.end_time(), next_obj.t - preempt)
        approach_t = frame_times[(frame_times >= start_time) & (frame_times < next_obj.t)]
        approach_f = (approach_t - start_time) / (next_obj.t - start_time)
        positions.extend((1 - approach_f[:, None]) * current_obj.end_pos() + approach_f[:, None] * next_obj.start_pos())

    return (np.array(positions) / np.array([512, 384])).T
