import random
from pathlib import Path
from typing import Dict, Generator, Tuple

import librosa
import numpy as np
import torch
from rosu_pp_py import Beatmap as RosuBeatmap
from rosu_pp_py import Difficulty as RosuDifficulty
from torch.utils.data import IterableDataset

from osu_fusion.library.augment import flip_cursor_horizontal, flip_cursor_vertical
from osu_fusion.library.osu.data.decode import Metadata, decode_beatmap
from osu_fusion.scripts.dataset_creator import (
    AUDIO_DIM,
    HOP_LENGTH,
    SR,
    normalize_context,
    unnormalize_context,
)


def load_tensor(map_file: Path, load_audio: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    map_data = np.load(map_file)
    audio_file = map_file.parent / map_data["spec_path"].tolist()
    audio_data = np.load(audio_file)
    x = torch.tensor(map_data["x"], dtype=torch.float32)
    c = torch.tensor(map_data["c"], dtype=torch.float32)

    if load_audio:
        a = torch.tensor(audio_data["a"], dtype=torch.float32)
    else:
        # Dummy audio tensor
        a = torch.zeros((AUDIO_DIM, x.shape[-1]), dtype=torch.float32)

    if torch.isnan(x).any() or torch.isnan(a).any() or torch.isnan(c).any():
        msg = "Invalid values in map file"
        raise ValueError(msg)

    return x, a, c


def get_new_context(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    cs, ar, od, hp, _ = unnormalize_context(
        c.clone(),  # Need to clone because unnormalize_context is in-place
    ).tolist()
    frame_times = (
        librosa.frames_to_time(
            np.arange(x.shape[-1]),
            sr=SR,
            hop_length=HOP_LENGTH,
        )
        * 1000
    )

    metadata = Metadata(
        "",
        "Dummy",
        "Dummy",
        "OsuFusion",
        cs,
        ar,
        od,
        hp,
    )
    segment_osu = decode_beatmap(metadata, x.numpy(), frame_times, bpm=None, allow_beat_snap=False, verbose=False)
    segment_beatmap = RosuBeatmap(content=segment_osu)
    rosu_difficulty = RosuDifficulty()
    segment_sr = rosu_difficulty.calculate(segment_beatmap).stars

    c = normalize_context(np.array([cs, ar, od, hp, segment_sr], dtype=np.float32))
    return torch.from_numpy(c)


class StreamPerSample(IterableDataset):
    def __init__(self: "StreamPerSample", **kwargs: Dict) -> None:
        super().__init__()
        self.dataset = kwargs.pop("dataset")
        self.sample_density = kwargs.pop("sample_density", 1.0)
        self.segment_sr = kwargs.pop("segment_sr", True)
        self.flip_horizontal_prob = kwargs.pop("flip_horizontal_prob", 0.5)
        self.flip_vertical_prob = kwargs.pop("flip_vertical_prob", 0.5)
        self.load_audio = kwargs.pop("load_audio", True)

        if not (0 < self.sample_density <= 1):
            msg = "sample_density must be between 0 and 1"
            raise ValueError(msg)

    def sample_stream(self: "StreamPerSample", map_file: Path) -> Generator[torch.Tensor, None, None]:
        raise NotImplementedError

    def __iter__(self: "StreamPerSample") -> Generator[torch.Tensor, None, None]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
            seed = torch.initial_seed()
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            seed = worker_info.seed

        random.seed(seed)

        for i, sample in random.sample(list(enumerate(self.dataset)), int(len(self.dataset) * self.sample_density)):
            if i % num_workers != worker_id:
                continue

            for x, a, c in self.sample_stream(sample):
                if self.segment_sr:
                    c = get_new_context(x, c)
                if random.random() < self.flip_horizontal_prob:
                    x = flip_cursor_horizontal(x)
                if random.random() < self.flip_vertical_prob:
                    x = flip_cursor_vertical(x)
                yield x, a, c

        # Randomize the dataset order for each epoch
        random.shuffle(self.dataset)


class FullSequenceDataset(StreamPerSample):
    MAX_LENGTH = 65536

    def sample_stream(self: StreamPerSample, map_file: Path) -> Generator[torch.Tensor, None, None]:
        x, a, c = load_tensor(map_file, load_audio=self.load_audio)

        if x.shape[-1] > self.MAX_LENGTH:
            return

        yield x[..., : self.MAX_LENGTH], a[..., : self.MAX_LENGTH], c


class SubsequenceDataset(StreamPerSample):
    def __init__(self: "SubsequenceDataset", **kwargs: Dict) -> None:
        super().__init__(**kwargs)
        self.sequence_length = kwargs.pop("sequence_length", 8192)  # 65.536 seconds

    def sample_stream(self: StreamPerSample, map_file: Path) -> Generator[torch.Tensor, None, None]:
        try:
            x, a, c = load_tensor(map_file, load_audio=self.load_audio)
        except ValueError:
            return
        n = x.shape[-1]

        if self.sequence_length > n:
            return

        # Random sampling
        start = random.randint(0, n - self.sequence_length)
        yield x[..., start : start + self.sequence_length], a[..., start : start + self.sequence_length], c
