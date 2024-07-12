import random
from pathlib import Path
from typing import Dict, Generator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from osu_fusion.library.osu.data.encode import TOTAL_DIM
from osu_fusion.scripts.dataset_creator import AUDIO_DIM, CONTEXT_DIM


def load_tensor(map_file: Path) -> torch.Tensor:
    map_data = np.load(map_file)
    audio_file = map_file.parent / map_data["spec_path"].tolist()
    audio_data = np.load(audio_file)
    x = torch.tensor(map_data["x"], dtype=torch.float32)
    c = torch.tensor(map_data["c"], dtype=torch.float32)
    a = torch.tensor(audio_data["a"], dtype=torch.float32)

    if torch.isnan(x).any() or torch.isnan(a).any() or torch.isnan(c).any():
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
            seed = torch.initial_seed()
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            seed = worker_info.seed

        random.seed(seed)

        for i, sample in random.sample(list(enumerate(self.dataset)), int(len(self.dataset) * self.sample_density)):
            if i % num_workers != worker_id:
                continue

            for x in self.sample_stream(sample):
                yield x

        # Randomize the dataset order for each epoch
        random.shuffle(self.dataset)


class DummyDataset(StreamPerSample):
    MIN_LENGTH = 4096
    MAX_LENGTH = 8192

    def sample_stream(self: StreamPerSample, _: Path) -> Generator[torch.Tensor, None, None]:
        length = random.randint(self.MIN_LENGTH, self.MAX_LENGTH)
        x = torch.randn((TOTAL_DIM, length))
        a = torch.randn((AUDIO_DIM, length))
        c = torch.randn((CONTEXT_DIM))

        yield x, a, c


class FullSequenceDataset(StreamPerSample):
    def sample_stream(self: StreamPerSample, map_file: Path) -> Generator[torch.Tensor, None, None]:
        yield load_tensor(map_file)


class SubsequenceDataset(StreamPerSample):
    def __init__(self: "SubsequenceDataset", **kwargs: Dict) -> None:
        super().__init__(**kwargs)

        self.sequence_length = kwargs.pop("sequence_length", 4096)
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
