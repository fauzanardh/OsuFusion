import numpy as np
import numpy.typing as npt


class Timed(object):
    def __init__(self: "Timed", t: int) -> None:
        self.t = t

    def __lt__(self: "Timed", other: "Timed") -> bool:
        return self.t < other.t


class TimingPoint(Timed):
    def __init__(
        self: "TimingPoint",
        t: int,
        beat_length: float,
        slider_multiplier: float,
        meter: int,
    ) -> None:
        super().__init__(t)
        self.beat_length = beat_length
        self.slider_multiplier = slider_multiplier
        self.meter = meter

    def __eq__(self: "TimingPoint", other: "TimingPoint") -> bool:
        return all(
            [
                self.t == other.t,
                self.beat_length == other.beat_length,
                self.slider_multiplier == other.slider_multiplier,
                self.meter == other.meter,
            ],
        )


class HitObject(Timed):
    def __init__(self: "HitObject", t: int, new_combo: bool) -> None:
        super().__init__(t)
        self.new_combo = new_combo

    def end_time(self: "HitObject") -> int:
        raise NotImplementedError

    def start_pos(self: "HitObject") -> npt.NDArray:
        raise NotImplementedError

    def end_pos(self: "HitObject") -> npt.NDArray:
        return self.start_pos()


class Circle(HitObject):
    def __init__(self: "Circle", t: int, new_combo: bool, x: int, y: int) -> None:
        super().__init__(t, new_combo)
        self.x = x
        self.y = y

    def end_time(self: "Circle") -> int:
        return self.t

    def start_pos(self: "Circle") -> npt.NDArray:
        return np.array([self.x, self.y], dtype=np.float32)


class Spinner(HitObject):
    def __init__(self: "Spinner", t: int, new_combo: bool, u: int) -> None:
        super().__init__(t, new_combo)
        self.u = u

    def end_time(self: "Spinner") -> int:
        return self.u

    def start_pos(self: "Spinner") -> npt.NDArray:
        return np.array([256, 192], dtype=np.float32)


class Slider(HitObject):
    def __init__(
        self: "Slider",
        t: int,
        beat_length: float,
        slider_multiplier: float,
        new_combo: bool,
        slides: int,
        length: float,
    ) -> None:
        super().__init__(t, new_combo)
        self.slides = slides
        self.length = length
        self.slider_multiplier = slider_multiplier
        self.slide_duration = length / (slider_multiplier * 100) * beat_length

    def end_time(self: "Slider") -> int:
        return int(self.t + self.slide_duration * self.slides)

    def lerp(self: "Slider", t: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    def velocity(self: "Slider", t: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    def start_pos(self: "Slider") -> npt.NDArray:
        return self.lerp(np.array([0], dtype=np.float32))[0]

    def end_pos(self: "Slider") -> npt.NDArray:
        return self.lerp(np.array([self.slides % 2], dtype=np.float32))[0]
