import numpy as np
import numpy.typing as npt

from osu_fusion.library.osu.beatmap import Beatmap
from osu_fusion.library.osu.hit_objects import Slider, Spinner


def cursor_signal(beatmap: Beatmap, frame_times: npt.NDArray) -> npt.NDArray:
    hitobject_iter = iter(beatmap.hit_objects)
    current_obj = None
    next_obj = next(hitobject_iter, None)

    positions = []

    for t in frame_times:
        while next_obj is not None and next_obj.t <= t:
            current_obj, next_obj = next_obj, next(hitobject_iter, None)

        if current_obj is None:
            if next_obj is None:
                positions.append(np.array([256, 192]))
            else:
                positions.append(next_obj.start_pos())
        elif t < current_obj.end_time():
            if isinstance(current_obj, Spinner):
                positions.append(current_obj.start_pos())
            elif isinstance(current_obj, Slider):
                ts = (t - current_obj.t) % (current_obj.slide_duration * 2) / current_obj.slide_duration
                if ts < 1:
                    positions.append(current_obj.lerp(ts))
                else:
                    positions.append(current_obj.lerp(2 - ts))
        elif next_obj is None:
            positions.append(current_obj.end_pos())
        else:
            f = (t - current_obj.end_time()) / (next_obj.t - current_obj.end_time())
            positions.append((1 - f) * current_obj.end_pos() + f * next_obj.start_pos())

    return (np.array(positions) / np.array([512, 384])).T
