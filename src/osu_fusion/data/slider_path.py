from typing import List, Optional

import numpy as np
import numpy.typing as npt
from numpy.linalg import norm

from osu_fusion.data import path_approximator


def binary_search(array: npt.NDArray, target: float) -> int:
    lower = 0
    upper = len(array)
    while lower < upper:  # use < instead of <=
        x = lower + (upper - lower) // 2
        val = array[x]
        if target == val:
            return x
        elif target > val:
            if lower == x:  # these two are the actual lines
                break  # you're looking for
            lower = x
        elif target < val:
            upper = x
    return ~upper


class SliderPath:
    def __init__(
        self,
        path_type: str,
        control_points: np.array,
        expected_distance: Optional[float] = None,
    ) -> None:
        self.control_points: npt.NDArray = control_points
        self.path_type: str = path_type
        self.expected_distance: Optional[float] = expected_distance

        self.calculated_path: List[npt.NDArray] = []
        self.cumulative_length: List[float] = []

        self.is_initialised: bool = False

        self.ensure_initialised()

    def get_control_points(self) -> npt.NDArray:
        self.ensure_initialised()
        return self.control_points

    def get_distance(self) -> float:
        self.ensure_initialised()
        return 0 if len(self.cumulative_length) == 0 else self.cumulative_length[-1]

    def get_path_to_progress(self, path: List[npt.NDArray], p0: float, p1: float) -> None:
        self.ensure_initialised()

        d0 = self.progress_to_distance(p0)
        d1 = self.progress_to_distance(p1)

        path.clear()

        i = 0
        while i < len(self.calculated_path) and self.cumulative_length[i] < d0:
            i += 1

        path.append(self.interpolate_vertices(i, d0))

        while i < len(self.calculated_path) and self.cumulative_length[i] < d1:
            path.append(self.calculated_path[i])
            i += 1

        path.append(self.interpolate_vertices(i, d1))

    def position_at(self, progress: float) -> npt.NDArray:
        self.ensure_initialised()

        d = self.progress_to_distance(progress)
        return self.interpolate_vertices(self.index_of_distance(d), d)

    def ensure_initialised(self) -> None:
        if self.is_initialised:
            return
        self.is_initialised = True

        self.control_points = [] if self.control_points is None else self.control_points
        self.calculated_path = []
        self.cumulative_length = []

        self.calculate_path()
        self.calculate_cumulative_length()

    def calculate_subpath(self, sub_control_points: npt.NDArray) -> list:
        if self.path_type == "L":
            return path_approximator.approximate_linear(sub_control_points)
        elif self.path_type == "P":
            if len(self.get_control_points()) != 3 or len(sub_control_points) != 3:
                return path_approximator.approximate_bezier(sub_control_points)

            subpath = path_approximator.approximate_circular_arc(sub_control_points)

            if len(subpath) == 0:
                return path_approximator.approximate_bezier(sub_control_points)

            return subpath
        elif self.path_type == "C":
            return path_approximator.approximate_catmull(sub_control_points)
        else:
            return path_approximator.approximate_bezier(sub_control_points)

    def calculate_path(self) -> None:
        self.calculated_path.clear()

        start = 0
        end = 0

        for i in range(len(self.get_control_points())):
            end += 1  # noqa: SIM113

            if (
                i == len(self.get_control_points()) - 1
                or (self.get_control_points()[i] == self.get_control_points()[i + 1]).all()
            ):
                cp_span = self.get_control_points()[start:end]

                for t in self.calculate_subpath(cp_span):
                    if len(self.calculated_path) == 0 or (self.calculated_path[-1] != t).any():
                        self.calculated_path.append(t)

                start = end

    def calculate_cumulative_length(self) -> None:
        length = 0

        self.cumulative_length.clear()
        self.cumulative_length.append(length)

        for i in range(len(self.calculated_path) - 1):
            diff = self.calculated_path[i + 1] - self.calculated_path[i]
            d = norm(diff)

            if self.expected_distance is not None and self.expected_distance - length < d:
                self.calculated_path[i + 1] = self.calculated_path[i] + diff * (self.expected_distance - length) / d
                del self.calculated_path[i + 2 : len(self.calculated_path) - 2 - i]

                length = self.expected_distance
                self.cumulative_length.append(length)
                break

            length += d
            self.cumulative_length.append(length)

        if self.expected_distance is not None and length < self.expected_distance and len(self.calculated_path) > 1:
            diff = self.calculated_path[-1] - self.calculated_path[-2]
            d = norm(diff)

            if d <= 0:
                return

            self.calculated_path[-1] += diff * (self.expected_distance - self.cumulative_length[-1]) / d
            self.cumulative_length[-1] = self.expected_distance

    def index_of_distance(self, d: float) -> int:
        i = binary_search(self.cumulative_length, d)
        if i < 0:
            i = ~i

        return i

    def progress_to_distance(self, progress: float) -> float:
        return np.clip(progress, 0, 1) * self.get_distance()

    def interpolate_vertices(self, i: int, d: float) -> npt.NDArray:
        if len(self.calculated_path) == 0:
            return np.zeros([2])

        if i <= 0:
            return self.calculated_path[0]
        if i >= len(self.calculated_path):
            return self.calculated_path[-1]

        p0 = self.calculated_path[i - 1]
        p1 = self.calculated_path[i]

        d0 = self.cumulative_length[i - 1]
        d1 = self.cumulative_length[i]

        if np.isclose(d0, d1):
            return p0

        w = (d - d0) / (d1 - d0)
        return p0 + (p1 - p0) * w


def position_to_progress(slider_path: "SliderPath", pos: npt.NDArray) -> npt.NDArray:
    eps = 1e-4
    lr = 1
    t = 1.0
    for _ in range(100):
        grad = np.linalg.norm(slider_path.position_at(t) - pos) - np.linalg.norm(
            slider_path.position_at(t - eps) - pos,
        )
        t -= lr * grad

        if grad == 0 or t < 0 or t > 1:
            break

    return np.clip(t, 0, 1)
