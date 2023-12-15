import datetime
import lzma
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class GameMode(IntEnum):
    STANDARD = 0
    TAIKO = 1
    CATCH = 2
    MANIA = 3


class Key(IntEnum):
    M1 = 1 << 0
    M2 = 1 << 1
    K1 = 1 << 2
    K2 = 1 << 3
    SMOKE = 1 << 4


@dataclass
class ReplayEventOsu:
    time_delta: int
    x: float
    y: float
    keys: Key


@dataclass
class LifeBarState:
    time: int
    life: float


class _Unpacker:
    def __init__(self: "_Unpacker", data: bytes) -> None:
        self.data = data
        self.offset = 0

    def string_length(self: "_Unpacker") -> int:
        out = 0
        shift = 0
        while True:
            b = self.data[self.offset]
            self.offset += 1
            out |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        return out

    def unpack_string(self: "_Unpacker") -> Optional[str]:
        if self.data[self.offset] in [0x0, 0x0B]:
            self.offset += 1
            if self.data[self.offset - 1] == 0x0:
                return None
            length = self.string_length()
            out = self.data[self.offset : self.offset + length].decode("utf-8")
            self.offset += length
            return out
        else:
            msg = "Invalid string type"
            raise ValueError(msg)

    def unpack_once(self: "_Unpacker", fmt: str) -> Tuple:
        specifier = f"<{fmt}"
        unpacked = struct.unpack_from(specifier, self.data, self.offset)
        self.offset += struct.calcsize(specifier)
        return unpacked[0]

    def unpack_timestamp(self: "_Unpacker") -> datetime.datetime:
        ticks = self.unpack_once("q")
        timestamp = datetime.datetime.min + datetime.timedelta(microseconds=ticks / 10)
        timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
        return timestamp

    @staticmethod
    def parse_replay_data(
        replay_data_str: str,
    ) -> Tuple[Optional[int], List[ReplayEventOsu]]:
        replay_data_str = replay_data_str.rstrip(",")
        events = [event.split("|") for event in replay_data_str.split(",")]

        rng_seed = None
        play_data = []
        for event in events:
            time_delta = int(event[0])
            x = float(event[1])
            y = float(event[2])
            keys = int(event[3])

            if time_delta == -12345 and event == events[-1]:
                rng_seed = keys
                continue

            play_data.append(ReplayEventOsu(time_delta, x, y, Key(keys)))

        return rng_seed, play_data

    def unpack_replay_data(self: "_Unpacker") -> Tuple[Optional[int], List[ReplayEventOsu]]:
        length = self.unpack_once("i")
        data = self.data[self.offset : self.offset + length]
        data = lzma.decompress(data, format=lzma.FORMAT_AUTO)
        data_str = data.decode("ascii")
        self.offset += length
        return self.parse_replay_data(data_str)

    def unpack_replay_id(self: "_Unpacker") -> int:
        try:
            return self.unpack_once("q")
        except struct.error:
            return self.unpack_once("l")

    def unpack_life_bar(self: "_Unpacker") -> Optional[List[LifeBarState]]:
        lifebar = self.unpack_string()
        if lifebar is None:
            return None

        lifebar = lifebar.rstrip(",")
        states = [state.split("|") for state in lifebar.split(",")]

        return [LifeBarState(int(state[0]), float(state[1])) for state in states]


class Replay:
    def __init__(self: "Replay", replay_path: str, to_np: bool = True) -> None:
        self._unpacker = _Unpacker(Path(replay_path).read_bytes())

        # Only used for osu!standard
        if GameMode(self._unpacker.unpack_once("b")) != GameMode.STANDARD:
            msg = "Invalid game mode"
            raise ValueError(msg)

        # Ignore most of the replay data
        self._unpacker.unpack_once("i")  # game_version
        self._unpacker.unpack_string()  # beatmap_hash
        self._unpacker.unpack_string()  # username
        self._unpacker.unpack_string()  # replay_hash
        self._unpacker.unpack_once("h")  # count_300
        self._unpacker.unpack_once("h")  # count_100
        self._unpacker.unpack_once("h")  # count_50
        self._unpacker.unpack_once("h")  # count_geki
        self._unpacker.unpack_once("h")  # count_katu
        self._unpacker.unpack_once("h")  # count_miss
        self._unpacker.unpack_once("i")  # score
        self._unpacker.unpack_once("h")  # max_combo
        self._unpacker.unpack_once("?")  # perfect
        self._unpacker.unpack_once("i")  # mods
        self._unpacker.unpack_life_bar()  # life_bar
        self._unpacker.unpack_timestamp()  # timestamp
        _, self._replay_data = self._unpacker.unpack_replay_data()  # rng_seed, replay_data
        self._unpacker.unpack_replay_id()  # replay_id

        del self._unpacker

        self.to_np = to_np
        if self.to_np:
            self.replay_data_to_np()

    def replay_data_to_np(self: "Replay") -> None:
        t = 0
        arr = np.zeros((len(self._replay_data), 3), dtype=np.float32)
        for i, event in enumerate(self._replay_data):
            t += event.time_delta
            arr[i] = [float(t), event.x, event.y]

        # Sort by time
        self._replay_data = arr[arr[:, 0].argsort()]

    def cursor(self: "Replay", t: int) -> Tuple[Tuple[float, float], float]:
        assert self.to_np, "Replay data is not in numpy format"

        idx = np.searchsorted(self._replay_data[:, 0], t, side="right") - 1
        if idx < 0:
            msg = f"Replay data does not contain any events before {t}"
            raise ValueError(msg)

        if idx == len(self._replay_data) - 1:
            return tuple(self._replay_data[idx, 1], self._replay_data[idx, 2]), 0.0

        t0, x0, y0 = self._replay_data[idx]
        t1, x1, y1 = self._replay_data[idx + 1]
        alpha = (t - t0) / (t1 - t0)
        return (x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)), t1 - t
