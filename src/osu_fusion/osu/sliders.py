from typing import List, Tuple

import bezier
import numpy as np
import numpy.typing as npt

from osu_fusion.osu.hit_objects import Slider

np.seterr(divide="raise")


class Line(Slider):
    def __init__(
        self: "Line",
        t: int,
        beat_length: float,
        slider_multiplier: float,
        new_combo: bool,
        slides: int,
        length: float,
        control_points: List[npt.NDArray],
    ) -> None:
        super().__init__(t, beat_length, slider_multiplier, new_combo, slides, length, control_points)

        self.path_segments, self.cum_t = self._calculate_path()

    def _calculate_path(self: "Line") -> Tuple[List[bezier.Curve], npt.NDArray]:
        total_length = 0
        curves = []
        for i in range(len(self.control_points) - 1):
            start, end = self.control_points[i], self.control_points[i + 1]
            nodes = np.array([start, end]).T
            line = bezier.Curve.from_nodes(nodes)
            total_length += line.length
            curves.append(line)

        tail_length = self.length - total_length
        if tail_length > 1e-3:
            last_point = self.control_points[-1]
            vec = last_point - self.control_points[-2]
            nodes = np.array([last_point, last_point + vec / np.linalg.norm(vec) * tail_length]).T
            tail = bezier.Curve.from_nodes(nodes)
            curves.append(tail)

        cum_t = np.cumsum([c.length for c in curves])
        return curves, cum_t / cum_t[-1]

    def curve_reparametrize(self: "Line", t: npt.NDArray) -> Tuple[int, npt.NDArray]:
        idx = np.searchsorted(self.cum_t, np.clip(t, 0, 1))
        range_start = np.insert(self.cum_t, 0, 0)[idx]
        range_end = self.cum_t[idx]
        return idx, (t - range_start) / (range_end - range_start)

    def lerp(self: "Line", t: npt.NDArray) -> npt.NDArray:
        return np.stack(
            [
                self.path_segments[idx].evaluate(t_val)[:, 0]
                for idx, t_val in zip(*self.curve_reparametrize(t), strict=True)
            ],
            axis=0,
        )

    def velocity(self: "Line", t: npt.NDArray) -> npt.NDArray:
        return np.stack(
            [
                self.path_segments[idx].evaluate_hodograph(t_val)[:, 0] / self.slide_duration
                for idx, t_val in zip(*self.curve_reparametrize(t), strict=True)
            ],
            axis=0,
        )


class Perfect(Slider):
    def __init__(
        self: "Perfect",
        t: int,
        beat_length: float,
        slider_multiplier: float,
        new_combo: bool,
        slides: int,
        length: float,
        control_points: List[npt.NDArray],
    ) -> None:
        super().__init__(t, beat_length, slider_multiplier, new_combo, slides, length, control_points)

        self.center, self.radius, self.start_angle, self.end_angle = self._calculate_circle()
        self.path_segments, self.cum_t = self._calculate_path()

    def _calculate_circle(self: "Perfect") -> Tuple[npt.NDArray, float, float, float]:
        pos1, pos2, pos3 = self.control_points
        a = np.linalg.norm(pos3 - pos2)
        b = np.linalg.norm(pos3 - pos1)
        c = np.linalg.norm(pos2 - pos1)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        radius = (a * b * c) / (4 * area)

        b1 = a * a * (b * b + c * c - a * a)
        b2 = b * b * (a * a + c * c - b * b)
        b3 = c * c * (a * a + b * b - c * c)
        center = np.column_stack((pos1, pos2, pos3)).dot(np.array([b1, b2, b3])) / (b1 + b2 + b3)

        start_angle = np.arctan2(*(pos1 - center)[[1, 0]])
        end_angle = np.arctan2(*(pos3 - center)[[1, 0]])

        if np.cross(pos2 - pos1, pos3 - pos1) < 0:
            while end_angle > start_angle:
                end_angle -= 2 * np.pi
        else:
            while start_angle > end_angle:
                start_angle -= 2 * np.pi
        return center, radius, start_angle, end_angle

    def _calculate_path(self: "Perfect") -> Tuple[List[bezier.Curve], npt.NDArray]:
        curves = []
        total_length = self.radius * abs(self.end_angle - self.start_angle)
        tail_length = self.length - total_length
        if tail_length > 1e-3:
            nodes = np.array(
                [
                    self.control_points[2],
                    self.control_points[2]
                    + (self.control_points[2] - self.control_points[1])
                    / np.linalg.norm(self.control_points[2] - self.control_points[1])
                    * tail_length,
                ],
            ).T
            curves.append(bezier.Curve.from_nodes(nodes))

        cum_t = np.cumsum([total_length] + [c.length for c in curves])
        return curves, cum_t / cum_t[-1]

    def _calculate_theta(self: "Perfect", t: float) -> float:
        return self.start_angle + t * (self.end_angle - self.start_angle)

    def lerp(self: "Perfect", t: npt.NDArray) -> npt.NDArray:
        if not self.path_segments:
            theta = self._calculate_theta(t)
            return self.center + self.radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)

        arc_t = self.cum_t[0]
        positions = []
        for t_val in t:
            if t_val <= arc_t:
                theta = self._calculate_theta(t_val / arc_t)
                positions.append(self.center + self.radius * np.array([np.cos(theta), np.sin(theta)]))
            else:
                tail_t = (t_val - arc_t) / (1 - arc_t)
                positions.append(self.path_segments[0].evaluate(tail_t)[:, 0])
        return np.array(positions)

    def velocity(self: "Perfect", t: npt.NDArray) -> npt.NDArray:
        if not self.path_segments:
            theta = self._calculate_theta(t)
            return (
                self.radius
                * np.stack([-np.sin(theta), np.cos(theta)], axis=1)
                * (self.end_angle - self.start_angle)
                / self.slide_duration
            )

        arc_t = self.cum_t[0]
        velocities = []
        for t_val in t:
            if t_val <= arc_t:
                theta = self._calculate_theta(t_val / arc_t)
                velocities.append(
                    self.radius
                    * np.array([-np.sin(theta), np.cos(theta)])
                    * (self.end_angle - self.start_angle)
                    / self.slide_duration,
                )
            else:
                tail_t = (t_val - arc_t) / (1 - arc_t)
                velocities.append(self.path_segments[0].evaluate_hodograph(tail_t)[:, 0] / self.slide_duration)
        return np.array(velocities)


class Bezier(Slider):
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
        super().__init__(t, beat_length, slider_multiplier, new_combo, slides, length, control_points)

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
        self.cum_t = np.cumsum([curve.length for curve in curves])
        self.cum_t /= self.cum_t[-1]

    def curve_reparametrize(self: "Bezier", t: npt.NDArray) -> Tuple[int, npt.NDArray]:
        idx = np.searchsorted(self.cum_t, np.clip(t, 0, 1))

        range_start = np.insert(self.cum_t, 0, 0)[idx]
        range_end = self.cum_t[idx]

        t = (t - range_start) / (range_end - range_start)
        return idx, t

    def lerp(self: "Bezier", t: npt.NDArray) -> npt.NDArray:
        return np.stack(
            [self.path_segments[idx].evaluate(t)[:, 0] for idx, t in zip(*self.curve_reparametrize(t), strict=True)],
            axis=0,
        )

    def velocity(self: "Bezier", t: npt.NDArray) -> npt.NDArray:
        return np.stack(
            [
                self.path_segments[idx].evaluate_hodograph(t)[:, 0] / self.slide_duration
                for idx, t in zip(*self.curve_reparametrize(t), strict=True)
            ],
            axis=0,
        )


def from_control_points(
    t: int,
    beat_length: float,
    slider_multiplier: float,
    new_combo: bool,
    slides: int,
    length: float,
    control_points: List[npt.NDArray],
    curve_type: str,
) -> Slider:
    if curve_type == "L":
        return Line(t, beat_length, slider_multiplier, new_combo, slides, length, control_points)
    if curve_type == "P":
        if len(control_points) > 3:
            return Bezier(t, beat_length, slider_multiplier, new_combo, slides, length, control_points)
        pos1, pos2, pos3 = control_points
        if np.abs(np.cross(pos2 - pos1, pos3 - pos1)) < 1e-3:
            return Line(t, beat_length, slider_multiplier, new_combo, slides, length, control_points)
        return Perfect(t, beat_length, slider_multiplier, new_combo, slides, length, control_points)
    if curve_type == "B":
        return Bezier(t, beat_length, slider_multiplier, new_combo, slides, length, control_points)
    if curve_type == "C":
        msg = "Catmull sliders are not implemented"
        raise NotImplementedError(msg)
    msg = f"Unknown curve type: {curve_type}"
    raise ValueError(msg)
