from typing import List, Tuple

import bezier
import numpy as np
import numpy.typing as npt

from osu_fusion.library.osu.hit_objects import Slider


def round_and_cast(value: npt.NDArray) -> npt.NDArray:
    return value.round(0).astype(int)


class Line(Slider):
    def __init__(
        self: "Line",
        t: int,
        beat_length: float,
        slider_multiplier: float,
        new_combo: bool,
        slides: int,
        length: float,
        start: npt.NDArray,
        end: npt.NDArray,
    ) -> None:
        super().__init__(t, beat_length, slider_multiplier, new_combo, slides, length)
        self.start = np.array(start)
        self.end = self._calculate_end(end, length)

    def _calculate_end(self: "Line", end: npt.NDArray, length: float) -> None:
        vec = end - self.start
        return self.start + vec / np.linalg.norm(vec) * length

    def lerp(self: "Line", t: float) -> npt.NDArray:
        return round_and_cast((1 - t) * self.start + t * self.end)

    def velocity(self: "Line", t: float) -> npt.NDArray:
        return round_and_cast((self.end - self.start) / self.slide_duration)


class Perfect(Slider):
    def __init__(
        self: "Perfect",
        t: int,
        beat_length: float,
        slider_multiplier: float,
        new_combo: bool,
        slides: int,
        length: float,
        center: npt.NDArray,
        radius: float,
        start: float,
        end: float,
    ) -> None:
        super().__init__(t, beat_length, slider_multiplier, new_combo, slides, length)
        self.center = np.array(center)
        self.radius = radius
        self.start = start
        self.end = self._calculate_end(start, length, radius, end)

    def _calculate_end(self: "Perfect", start: float, length: float, radius: float, end: float) -> float:
        return start + length / radius * np.sign(end - start)

    def _calculate_theta(self: "Perfect", t: float) -> float:
        return (1 - t) * self.start + t * self.end

    def lerp(self: "Perfect", t: float) -> npt.NDArray:
        theta = self._calculate_theta(t)
        return round_and_cast(self.center + self.radius * np.array([np.cos(theta), np.sin(theta)]))

    def velocity(self: "Perfect", t: float) -> npt.NDArray:
        theta = self._calculate_theta(t)
        return round_and_cast(
            self.radius * np.array([-np.sin(theta), np.cos(theta)]) / self.slide_duration,
        )


class Bezier(Slider):
    SEGMENTS = 10

    def __init__(
        self: "Bezier",
        t: int,
        beat_length: float,
        slider_multiplier: float,
        new_combo: bool,
        slides: int,
        length: float,
        control_points: List[npt.NDArray],
    ) -> None:
        super().__init__(t, beat_length, slider_multiplier, new_combo, slides, length)
        self.control_points = control_points

        control_curves = []
        last_idx = 0
        for i, point in enumerate(control_points[1:]):
            if (control_points[i] == point).all():
                control_curves.append(control_points[last_idx : i + 1])
                last_idx = i + 1
        control_curves.append(control_points[last_idx:])

        total_length = 0
        curves = []
        for curve in control_curves:
            if len(curve) < 2:
                continue

            nodes = np.array(curve).T
            bezier_curve = bezier.Curve.from_nodes(nodes)
            total_length += bezier_curve.length
            curves.append(bezier_curve)

        tail_length = self.length - total_length
        if tail_length > 0:
            last_curve_nodes = curves[-1].nodes
            point = last_curve_nodes[:, -1]
            vec = point - last_curve_nodes[:, -2]

            nodes = np.array([point, point + vec / np.linalg.norm(vec) * tail_length]).T
            bezier_curve = bezier.Curve.from_nodes(nodes)

            assert np.isclose(bezier_curve.length, tail_length), f"{bezier_curve.length} != {tail_length}"
            curves.append(bezier_curve)

        self.path_segments = curves
        self.cum_t = np.cumsum([curve.length for curve in curves]) / self.length
        self.cum_t[-1] = 1.0

    def curve_reparametrize(self: "Bezier", t: float) -> Tuple[int, float]:
        idx = np.searchsorted(self.cum_t, min(1, max(0, t)))
        assert idx < len(self.cum_t), f"{idx} >= {len(self.cum_t)}"

        range_start = np.insert(self.cum_t, 0, 0)[idx]
        range_end = self.cum_t[idx]

        t = (t - range_start) / (range_end - range_start)
        return int(idx), t

    def lerp(self: "Bezier", t: float) -> npt.NDArray:
        idx, t = self.curve_reparametrize(t)
        return round_and_cast(self.path_segments[idx].evaluate(t)[:, 0])

    def velocity(self: "Bezier", t: float) -> npt.NDArray:
        idx, t = self.curve_reparametrize(t)
        return round_and_cast(
            self.path_segments[idx].evaluate_hodograph(t)[:, 0] / self.slide_duration,
        )


def from_control_points(
    t: int,
    beat_length: float,
    slider_multiplier: float,
    new_combo: bool,
    slides: int,
    length: float,
    control_points: List[npt.NDArray],
) -> Slider:
    assert len(control_points) >= 2, f"not enough control points: {len(control_points)}"

    if len(control_points) == 2:  # Line
        pos1, pos2 = control_points
        return Line(t, beat_length, slider_multiplier, new_combo, slides, length, pos1, pos2)
    if len(control_points) == 3:  # Perfect/Line/Bezier
        pos1, pos2, pos3 = control_points

        if (pos2 == pos3).all():
            return Line(t, beat_length, slider_multiplier, new_combo, slides, length, pos1, pos3)

        cross_product = np.cross(pos2 - pos1, pos3 - pos1)
        if cross_product == 0:  # Collinear
            if np.dot(pos2 - pos1, pos3 - pos1) > 0:
                return Line(t, beat_length, slider_multiplier, new_combo, slides, length, pos1, pos3)
            else:
                control_points.insert(1, control_points[1])
                return Bezier(t, beat_length, slider_multiplier, new_combo, slides, length, control_points)

        a = np.linalg.norm(pos3 - pos2)
        b = np.linalg.norm(pos3 - pos1)
        c = np.linalg.norm(pos2 - pos1)
        s = (a + b + c) / 2
        r = a * b * c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))

        if r > 320 and np.dot(pos3 - pos2, pos2 - pos1) > 0:
            return Bezier(t, beat_length, slider_multiplier, new_combo, slides, length, control_points)

        b1 = a * a * (b * b + c * c - a * a)
        b2 = b * b * (a * a + c * c - b * b)
        b3 = c * c * (a * a + b * b - c * c)
        p = np.column_stack((pos1, pos2, pos3)).dot(np.hstack((b1, b2, b3)))
        p /= b1 + b2 + b3

        start_angle = np.arctan2(*(pos1 - p)[[1, 0]])
        end_angle = np.arctan2(*(pos3 - p)[[1, 0]])

        if cross_product < 0:  # Clockwise
            while end_angle > start_angle:
                end_angle -= 2 * np.pi
        else:  # Counter-clockwise
            while start_angle > end_angle:
                start_angle -= 2 * np.pi

        return Perfect(t, beat_length, slider_multiplier, new_combo, slides, length, p, r, start_angle, end_angle)
    else:  # Bezier
        return Bezier(t, beat_length, slider_multiplier, new_combo, slides, length, control_points)
