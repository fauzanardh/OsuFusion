import random
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

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


class BeatmapDataset(Dataset):
    def __init__(self: "BeatmapDataset", **kwargs: dict) -> None:
        super().__init__()
        self.dataset = kwargs.pop("dataset")
        self.flip_horizontal_prob = kwargs.pop("flip_horizontal_prob", 0.5)
        self.flip_vertical_prob = kwargs.pop("flip_vertical_prob", 0.5)
        self.load_audio = kwargs.pop("load_audio", True)

        self.tensor_loader = TensorLoader()

    def __len__(self: "BeatmapDataset") -> int:
        return len(self.dataset)

    def __getitem__(self: "BeatmapDataset", index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        map_data = self.tensor_loader.load_tensor(self.dataset[index], self.load_audio)
        x, a, c = map_data.x, map_data.a, map_data.c

        if random.random() < self.flip_horizontal_prob:
            x = flip_cursor_horizontal(x)
        if random.random() < self.flip_vertical_prob:
            x = flip_cursor_vertical(x)

        return x, a, c
