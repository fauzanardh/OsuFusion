from typing import List

import numpy as np
import numpy.typing as npt
from slider.beatmap import Beatmap, Circle, Slider, Spinner
from slider.curve import Catmull, Linear, MultiBezier, Perfect

from osu_fusion.data.event import EventType


def create_datapoint(time: float, pos: tuple[float, float], datatype: int) -> npt.NDArray:
    features = np.zeros(19)
    features[0] = max(-256, min(768, pos[0]))
    features[1] = max(-256, min(640, pos[1]))
    features[2] = time
    features[datatype + 3] = 1
    return features


def repeat_type(repeat: int) -> int:
    if repeat < 4:
        return repeat - 1
    elif repeat % 2 == 0:
        return 3
    else:
        return 4


def append_control_points(
    datapoints: List[npt.NDArray],
    slider: Slider,
    datatype: int,
    duration: float,
) -> None:
    control_point_count = len(slider.curve.points)
    start_time = slider.time.total_seconds() * 1000

    for i in range(1, control_point_count - 1):
        time = start_time + i / (control_point_count - 1) * duration
        pos = slider.curve.points[i]
        datapoints.append(create_datapoint(time, pos, datatype))


def get_data(hitobj: Circle | Slider | Spinner) -> npt.NDArray:
    if isinstance(hitobj, Slider):
        datapoints = [
            create_datapoint(
                hitobj.time.total_seconds() * 1000,
                hitobj.position,
                EventType.SLIDER_HEAD_NEW_COMBO if hitobj.new_combo else EventType.SLIDER_HEAD,
            ),
        ]

        assert hitobj.repeat >= 1
        duration = (hitobj.end_time - hitobj.time).total_seconds() * 1000 / hitobj.repeat

        if isinstance(hitobj.curve, Linear):
            append_control_points(datapoints, hitobj, EventType.LINEAR_ANCHOR, duration)
        elif isinstance(hitobj.curve, Catmull):
            append_control_points(datapoints, hitobj, EventType.CATMULL_ANCHOR, duration)
        elif isinstance(hitobj.curve, Perfect):
            append_control_points(datapoints, hitobj, EventType.PERFECT_ANCHOR, duration)
        elif isinstance(hitobj.curve, MultiBezier):
            control_point_count = len(hitobj.curve.points)
            start_time = hitobj.time.total_seconds() * 1000

            for i in range(1, control_point_count - 1):
                time = start_time + i / (control_point_count - 1) * duration
                pos = hitobj.curve.points[i]

                if pos == hitobj.curve.points[i + 1]:
                    datapoints.append(create_datapoint(time, pos, EventType.LINEAR_ANCHOR))
                elif pos != hitobj.curve.points[i - 1]:
                    datapoints.append(create_datapoint(time, pos, EventType.BEZIER_ANCHOR))

        datapoints.append(
            create_datapoint(
                hitobj.time.total_seconds() * 1000 + duration,
                hitobj.curve.points[-1],
                EventType.LAST_ANCHOR,
            ),
        )

        slider_end_pos = hitobj.curve(1)
        datapoints.append(
            create_datapoint(
                hitobj.end_time.total_seconds() * 1000,
                slider_end_pos,
                EventType.SLIDER_END_REPEAT_1 + repeat_type(hitobj.repeat),
            ),
        )

        return np.stack(datapoints, 0)

    if isinstance(hitobj, Spinner):
        return np.stack(
            (
                create_datapoint(
                    hitobj.time.total_seconds() * 1000,
                    hitobj.position,
                    EventType.SPINNER_START,
                ),
                create_datapoint(
                    hitobj.end_time.total_seconds() * 1000,
                    hitobj.position,
                    EventType.SPINNER_END,
                ),
            ),
            0,
        )

    return create_datapoint(
        hitobj.time.total_seconds() * 1000,
        hitobj.position,
        EventType.CIRCLE_NEW_COMBO if hitobj.new_combo else EventType.CIRCLE,
    )[None]


def encode_beatmap(beatmap: Beatmap) -> npt.NDArray:
    hit_objects = beatmap.hit_objects(stacking=False)
    data_chunks = [get_data(ho) for ho in hit_objects]

    sequence = np.concatenate(data_chunks, 0)
    sequence = np.swapaxes(sequence, 0, 1)

    return sequence.astype(np.float32)
