import numpy as np
import numpy.typing as npt
from slider.beatmap import Beatmap, Circle, Slider, Spinner


def cursor_signal(beatmap: Beatmap, frame_times: npt.NDArray) -> npt.NDArray:  # noqa: C901
    if beatmap.approach_rate < 5:
        preempt = 1200 + 600 * (5 - beatmap.approach_rate) / 5
    elif beatmap.approach_rate == 5:
        preempt = 1200
    else:
        preempt = 1200 - 750 * (beatmap.approach_rate - 5) / 5

    hit_objects = beatmap.hit_objects()
    positions = []

    # Manually add a starting circle at the beginning of the map
    if hit_objects:
        first_obj_time = hit_objects[0].time.total_seconds() * 1000
        start_pos = np.array([256, 192])
        initial_approach_t = frame_times[frame_times < first_obj_time]
        if len(initial_approach_t) > 0:
            start_time = max(0, first_obj_time - preempt)
            approach_f = (initial_approach_t - start_time) / (first_obj_time - start_time)
            positions.extend(
                (1 - approach_f[:, None]) * start_pos
                + approach_f[:, None] * np.array([hit_objects[0].position.x, hit_objects[0].position.y]),
            )

    for i, current_obj in enumerate(hit_objects):
        next_obj = hit_objects[i + 1] if i + 1 < len(hit_objects) else None
        current_obj_start_time = current_obj.time.total_seconds() * 1000
        current_obj_end_time = (
            current_obj.end_time.total_seconds() * 1000
            if isinstance(current_obj, (Slider, Spinner))
            else current_obj_start_time
        )

        if isinstance(current_obj, Spinner):
            current_count = np.sum((frame_times >= current_obj_start_time) & (frame_times < current_obj_end_time))
            positions.extend(np.array([256, 192])[None].repeat(current_count, axis=0))
        elif isinstance(current_obj, Slider):
            current_t = frame_times[(frame_times >= current_obj_start_time) & (frame_times < current_obj_end_time)]
            slide_duration = (
                (current_obj.end_time.total_seconds() - current_obj.time.total_seconds()) * 1000 / current_obj.repeat
            )
            current_f = (current_t - current_obj_start_time) % (slide_duration * 2) / slide_duration
            if len(current_f) > 0:
                progress = np.where(current_f < 1, current_f, 2 - current_f)
                positions.extend([current_obj.curve(p) for p in progress])

        if next_obj is None:
            map_end_count = np.sum(frame_times >= current_obj_end_time)
            end_pos = (
                np.array([current_obj.position.x, current_obj.position.y])
                if isinstance(current_obj, Circle)
                else np.array(current_obj.curve(current_obj.repeat % 2))
                if isinstance(current_obj, Slider)
                else np.array([256, 192])
            )
            positions.extend(end_pos[None].repeat(map_end_count, axis=0))
            break

        next_obj_start_time = next_obj.time.total_seconds() * 1000
        wait_count = np.sum((frame_times >= current_obj_end_time) & (frame_times < next_obj_start_time - preempt))
        end_pos = (
            np.array([current_obj.position.x, current_obj.position.y])
            if isinstance(current_obj, Circle)
            else np.array(current_obj.curve(current_obj.repeat % 2))
            if isinstance(current_obj, Slider)
            else np.array([256, 192])
        )
        positions.extend(end_pos[None].repeat(wait_count, axis=0))

        start_time = max(current_obj_end_time, next_obj_start_time - preempt)
        approach_t = frame_times[(frame_times >= start_time) & (frame_times < next_obj_start_time)]
        if len(approach_t) > 0:
            approach_f = (approach_t - start_time) / (next_obj_start_time - start_time)
            next_start_pos = np.array([next_obj.position.x, next_obj.position.y])
            positions.extend((1 - approach_f[:, None]) * end_pos + approach_f[:, None] * next_start_pos)

    return np.clip((np.array(positions) / np.array([512, 384])).T, -1, 1)
