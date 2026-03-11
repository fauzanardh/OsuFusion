import random
from pathlib import Path
from typing import Generator, NamedTuple, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset

from osu_fusion.data.augment import flip_cursor_horizontal, flip_cursor_vertical
from osu_fusion.data.const import AUDIO_DIM


class MapData(NamedTuple):
    x: torch.Tensor
    a: torch.Tensor
    c: torch.Tensor
    spec_path: str


class TensorLoader:
    def __init__(self: "TensorLoader", device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cpu")

    def _to_tensor(self: "TensorLoader", array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(device=self.device, dtype=torch.float32)

    def load_tensor(self: "TensorLoader", map_file: Path, load_audio: bool = True) -> MapData:
        with h5py.File(map_file, "r") as map_data:
            x = self._to_tensor(map_data["x"][:])
            c = self._to_tensor(map_data["c"][:])
            spec_path = map_data["spec_path"][()].decode("utf-8")

        if load_audio:
            audio_file = map_file.parent.parent.parent / spec_path
            with h5py.File(audio_file, "r") as audio_data:
                a = self._to_tensor(audio_data["a"][:])
        else:
            a = torch.zeros((x.shape[0], AUDIO_DIM), dtype=torch.float32)

        if any(torch.isnan(t).any() for t in [x, a, c]):
            msg = "Invalid values in map file"
            raise ValueError(msg)

        return MapData(
            x=x,
            a=a,
            c=c,
            spec_path=spec_path,
        )


class BeatmapDataset(IterableDataset):
    def __init__(self: "BeatmapDataset", **kwargs: dict) -> None:
        super().__init__()
        self.dataset = kwargs.pop("dataset")
        self.sample_density = kwargs.pop("sample_density", 1.0)
        self.flip_horizontal_prob = kwargs.pop("flip_horizontal_prob", 0.5)
        self.flip_vertical_prob = kwargs.pop("flip_vertical_prob", 0.5)
        self.load_audio = kwargs.pop("load_audio", True)

        self.tensor_loader = TensorLoader()

        if not (0 < self.sample_density <= 1):
            msg = "sample_density must be between 0 and 1"
            raise ValueError(msg)

    def __iter__(self: "BeatmapDataset") -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        worker_info = torch.utils.data.get_worker_info()
        indices = list(range(len(self.dataset)))

        if self.sample_density < 1.0:
            num_samples = int(len(indices) * self.sample_density)
            indices = random.sample(indices, num_samples)

        random.shuffle(indices)
        if worker_info is None:
            indices_for_worker = indices
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            indices_for_worker = indices[worker_id::num_workers]

        for index in indices_for_worker:
            map_file = self.dataset[index]
            try:
                map_data = self.tensor_loader.load_tensor(map_file, self.load_audio)
                x, a, c = map_data.x, map_data.a, map_data.c

                if random.random() < self.flip_horizontal_prob:
                    x = flip_cursor_horizontal(x)
                if random.random() < self.flip_vertical_prob:
                    x = flip_cursor_vertical(x)

                yield x, a, c
            except Exception as e:
                print(f"Error processing sample {map_file}: {e}")
                continue
