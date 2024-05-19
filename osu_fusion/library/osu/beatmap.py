import bisect
import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import numpy as np

from osu_fusion.library.osu.hit_objects import Circle, HitObject, Slider, Spinner, Timed, TimingPoint
from osu_fusion.library.osu.sliders import from_control_points

CX, CY = 256, 192


class Beatmap:
    def __init__(self: "Beatmap", filename: Path, meta_only: bool = False) -> None:
        self.filename = filename

        self.timing_points = []
        self.uninherited_timing_points = []
        self.hit_objects = []
        self.events = []

        with open(self.filename, "r", encoding="utf-8") as f:
            cfg = self.parse_beatmap(f.readlines())

        self.audio_filename = self.filename.parent / cfg["General"]["AudioFilename"]

        self.mode = int(cfg["General"]["Mode"])

        self.title = cfg["Metadata"]["Title"]
        self.artist = cfg["Metadata"]["Artist"]
        self.creator = cfg["Metadata"]["Creator"]
        self.version = cfg["Metadata"]["Version"]

        try:
            self.mapset_id = int(cfg["Metadata"]["BeatmapSetID"])
        except KeyError:
            self.mapset_id = None

        self.hp = float(cfg["Difficulty"]["HPDrainRate"])
        self.cs = float(cfg["Difficulty"]["CircleSize"])
        self.od = float(cfg["Difficulty"]["OverallDifficulty"])
        try:
            self.ar = float(cfg["Difficulty"]["ApproachRate"])
        except KeyError:
            self.ar = 7

        self.slider_multiplier = float(cfg["Difficulty"]["SliderMultiplier"])
        self.slider_tick_rate = float(cfg["Difficulty"]["SliderTickRate"])

        try:
            self.beat_divisor = int(cfg["Difficulty"]["BeatDivisor"])
        except KeyError:
            self.beat_divisor = 4

        self.unparsed_hit_objects = cfg["HitObjects"]
        self.unparsed_timing_points = cfg["TimingPoints"]
        self.unparsed_events = cfg["Events"]

        if not meta_only:
            self.parse_map_data()

    def parse_timing_point(self: "Beatmap", lines: List[str]) -> None:
        cur_beat_length = None
        cur_slider_multiplier = 1.0
        cur_meter = None

        for line in lines:
            vals = [float(x) for x in line.strip().split(",")]
            t, x, meter = vals[:3]
            kiai = int(vals[7] if len(vals) >= 8 else 0) % 2 == 1

            if vals[6] == 0:
                if len(self.timing_points) == 0:
                    continue

                if self.timing_points[-1].t == t:
                    self.timing_points.pop()

                cur_slider_multiplier = min(10.0, max(0.1, round(-100 / float(x), 3)))
            else:
                cur_beat_length = x
                cur_slider_multiplier = 1.0
                cur_meter = meter

            tp = TimingPoint(int(t), cur_beat_length, cur_slider_multiplier, cur_meter, kiai)
            if len(self.timing_points) == 0 or tp != self.timing_points[-1]:
                self.timing_points.append(tp)

            utp = TimingPoint(int(t), cur_beat_length, None, cur_meter, None)
            if len(self.uninherited_timing_points) == 0 or utp != self.uninherited_timing_points[-1]:
                self.uninherited_timing_points.append(utp)

        if len(self.timing_points) == 0:
            msg = "no timing points found"
            raise ValueError(msg)

    def get_active_timing_point(self: "Beatmap", t: int) -> TimingPoint:
        idx = bisect.bisect(self.timing_points, Timed(t)) - 1
        if idx < 0:
            msg = f"no active timing point at {t}"
            raise ValueError(msg)

        return self.timing_points[idx]

    def parse_hit_object(self: "Beatmap", lines: List[str]) -> None:
        for line in lines:
            vals = line.strip().split(",")
            x, y, t, k = [int(x) for x in vals[:4]]
            new_combo = (k & (1 << 2)) > 0
            if k & (1 << 0):
                ho = Circle(t, new_combo, x, y)
            elif k & (1 << 1):
                curve, slides, length = vals[5:8]
                _, *control_points = curve.split("|")
                control_points = [np.array([x, y])] + [np.array(list(map(int, p.split(":")))) for p in control_points]

                tp = self.get_active_timing_point(t)
                ho = from_control_points(
                    t,
                    tp.beat_length,
                    self.slider_multiplier * tp.slider_multiplier,
                    new_combo,
                    int(slides),
                    float(length),
                    control_points,
                )
            elif k & (1 << 3):
                ho = Spinner(t, new_combo, int(vals[5]))

            if len(self.hit_objects) and ho.t < self.hit_objects[-1].end_time():
                msg = f"hit objects not in chronological order: {ho.t} < {self.hit_objects[-1].end_time()}"
                raise ValueError(msg)

            self.hit_objects.append(ho)

        if len(self.hit_objects) == 0:
            msg = "no hit objects found"
            raise ValueError(msg)

    def parse_events(self: "Beatmap", lines: List[str]) -> None:
        self.events = []
        for line in lines:
            vals = line.strip().split(",")
            if vals[0] == 2:
                self.events.append(vals)

    def parse_map_data(self: "Beatmap") -> None:
        self.parse_timing_point(self.unparsed_timing_points)
        del self.unparsed_timing_points
        self.parse_hit_object(self.unparsed_hit_objects)
        del self.unparsed_hit_objects
        self.parse_events(self.unparsed_events)
        del self.unparsed_events

    @staticmethod
    def _process_circle_cursor(ho: Circle, nho: HitObject, t: int) -> Tuple[Tuple[int, int], float]:
        if nho is not None:
            f = t / (nho.t - ho.t)
            return ((1 - f) * ho.x + f * nho.x, (1 - f) * ho.y + f * nho.y), t
        else:
            return (ho.x, ho.y), t

    @staticmethod
    def _process_spinner_cursor(ho: Spinner, nho: HitObject, t: int) -> Tuple[Tuple[int, int], float]:
        spin_duration = ho.u - ho.t
        if t < spin_duration:
            return (CX, CY), 0
        else:
            t -= spin_duration
            if nho is not None:
                f = t / (nho.t - ho.t - spin_duration)
                return ((1 - f) * CX + f * nho.x, (1 - f) * CY + f * nho.y), t
            else:
                return (CX, CY), t

    @staticmethod
    def _process_slider_cursor(ho: Slider, nho: HitObject, t: int) -> Tuple[Tuple[int, int], float]:
        slide_duration = ho.slide_duration
        if t < slide_duration:
            single_slide_duration = slide_duration / ho.slides

            ts = t % (single_slide_duration * 2)
            if ts < single_slide_duration:  # forward
                return ho.lerp(ts / single_slide_duration), 0
            else:  # backward
                return ho.lerp(2 - ts / single_slide_duration), 0
        else:
            t -= slide_duration
            end = ho.lerp(ho.slides % 2)

            if nho is not None:
                f = t / (nho.t - ho.t - slide_duration)
                return ((1 - f) * end[0] + f * nho.x, (1 - f) * end[1] + f * nho.y), t
            else:
                return end, t

    def cursor(self: "Beatmap", t: int) -> Tuple[Tuple[int, int], float]:
        if t < self.hit_objects[0].t:
            ho = self.hit_objects[0]
            if isinstance(ho, Circle):
                return (ho.x, ho.y), np.inf
            elif isinstance(ho, Spinner):
                return (CX, CY), np.inf
            elif isinstance(ho, Slider):
                return ho.start_pos(), np.inf

        for ho, nho in zip(self.hit_objects, self.hit_objects[1:]):
            if ho.t <= t < nho.t:
                break
        else:
            ho = self.hit_objects[-1]
            nho = None

        if isinstance(ho, Circle):
            return Beatmap._process_circle_cursor(ho, nho, t)
        elif isinstance(ho, Spinner):
            return Beatmap._process_spinner_cursor(ho, nho, t)
        elif isinstance(ho, Slider):
            return Beatmap._process_slider_cursor(ho, nho, t)

    @staticmethod
    def parse_beatmap(lines: List[str]) -> Dict[str, Any]:
        list_sections = ["Events", "TimingPoints", "HitObjects"]
        cfg = {}
        section = None

        for line in lines:
            if line.startswith("//"):
                continue

            if line.strip() == "":
                section = None
                continue

            m = re.search(r"^\[(.*)\]$", line)
            if m is not None:
                section = m.group(1)
                if section in list_sections:
                    cfg[section] = []
                else:
                    cfg[section] = {}
                continue

            if section is None:
                continue

            if section in list_sections:
                cfg[section].append(line.strip())
            else:
                m2 = re.search(r"^(\w*)\s?:\s?(.*)$", line)
                if m2 is not None:
                    cfg[section][m2.group(1)] = m2.group(2).strip()

        return cfg

    @staticmethod
    def all_maps(src_path: str, meta_only: bool = False) -> Generator["Beatmap", None, None]:
        path = Path(src_path)
        for filename in path.glob("*/*.osu"):
            try:
                beatmap = Beatmap(filename, meta_only=meta_only)
            except Exception as e:
                print(f"Failed to parse {filename}: {e}")
                continue

            if beatmap.mode != 0:
                continue

            yield beatmap

    @staticmethod
    def all_mapsets(src_path: str, meta_only: bool = False) -> Generator[Tuple[int, str, List["Beatmap"]], None, None]:
        mapset_path = Path(src_path)
        for mapset_dir in mapset_path.iterdir():
            if not mapset_dir.is_dir():
                continue

            maps = []
            mapset_id = None
            audio_file = None
            for map_file in mapset_dir.glob("*.osu"):
                try:
                    beatmap = Beatmap(map_file, meta_only=meta_only)
                except Exception as e:
                    print(f"Failed to parse {map_file}: {e}")
                    continue

                if beatmap.mode != 0:
                    continue

                if audio_file is None:
                    audio_file = beatmap.audio_filename

                if mapset_id is None:
                    mapset_id = beatmap.mapset_id
            else:
                if audio_file is None or mapset_id is None or len(maps) == 0:
                    continue
                yield mapset_id, audio_file, maps
