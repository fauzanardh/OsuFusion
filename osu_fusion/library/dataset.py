import random
from pathlib import Path
from typing import Dict, Generator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from osu_fusion.library.osu.from_beatmap import AUDIO_DIM, CONTEXT_DIM, TOTAL_DIM

MFCC_MAX_VALUE = 300
MFCC_MIN_VALUE = -600


def sanitize_input(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(min=-1.0, max=1.0)


# Temporary function to normalize MFCCs in my dataset
# TODO: Implement this to the dataset creator instead
def normalize_mfcc(mfcc: torch.Tensor) -> torch.Tensor:
    return (mfcc - MFCC_MIN_VALUE) / (MFCC_MAX_VALUE - MFCC_MIN_VALUE) * 2 - 1


def load_tensor(map_file: Path) -> torch.Tensor:
    audio_file = map_file.parent / "audio_mp3" / "spec.npz"

    map_data = np.load(map_file)
    audio_data = np.load(audio_file)
    x = torch.tensor(map_data["x"], dtype=torch.float32)
    c = torch.tensor(map_data["c"], dtype=torch.float32)
    a = torch.tensor(audio_data["a"], dtype=torch.float32)

    x = sanitize_input(x)
    a = sanitize_input(normalize_mfcc(a))
    c = sanitize_input(c)

    # Check if x a and c values are within the range of -1 to 1 and there are no NaN values
    if (
        x.max() > 1
        or x.min() < -1
        or torch.isnan(x).any()
        or a.max() > 1
        or a.min() < -1
        or torch.isnan(a).any()
        or c.max() > 1
        or c.min() < -1
        or torch.isnan(c).any()
    ):
        msg = "Invalid values in map file"
        raise ValueError(msg)

    return x, a, c


class StreamPerSample(IterableDataset):
    def __init__(self: "StreamPerSample", **kwargs: Dict) -> None:
        super().__init__()

        self.dataset = kwargs.pop("dataset")
        self.sample_density = kwargs.pop("sample_density", 1.0)

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
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        for i, sample in random.sample(list(enumerate(self.dataset)), int(len(self.dataset) * self.sample_density)):
            if i % num_workers != worker_id:
                continue

            try:
                for x in self.sample_stream(sample):
                    yield x
            except Exception:
                continue

        # Randomize the dataset order for each epoch
        random.shuffle(self.dataset)


class DummyDataset(StreamPerSample):
    MIN_LENGTH = 1024
    MAX_LENGTH = 4096

    def sample_stream(self: StreamPerSample, _: Path) -> Generator[torch.Tensor, None, None]:
        length = random.randint(self.MIN_LENGTH, self.MAX_LENGTH)
        x = torch.randn((TOTAL_DIM, length))
        a = torch.randn((AUDIO_DIM, length))
        c = torch.randn((CONTEXT_DIM))

        yield x, a, c


class FullSequenceDataset(StreamPerSample):
    MAX_LENGTH = 65536

    def sample_stream(self: StreamPerSample, map_file: Path) -> Generator[torch.Tensor, None, None]:
        x, a, c = load_tensor(map_file)

        if x.shape[-1] > self.MAX_LENGTH:
            return

        yield x[..., : self.MAX_LENGTH], a[..., : self.MAX_LENGTH], c


class SubsequenceDataset(StreamPerSample):
    def __init__(self: "SubsequenceDataset", **kwargs: Dict) -> None:
        super().__init__(**kwargs)

        self.sequence_length = kwargs.pop("sequence_length")
        self.subsequence_density = kwargs.pop("subsequence_density", 2.0)
        super().__init__(**kwargs)

    def sample_stream(self: StreamPerSample, map_file: Path) -> Generator[torch.Tensor, None, None]:
        try:
            x, a, c = load_tensor(map_file)
        except ValueError:
            return
        n = x.shape[-1]

        if self.sequence_length > n:
            return

        num_samples = int(n / self.sequence_length * self.subsequence_density)
        for i in torch.randperm(n - self.sequence_length)[:num_samples]:
            yield x[..., i : i + self.sequence_length], a[..., i : i + self.sequence_length], c
