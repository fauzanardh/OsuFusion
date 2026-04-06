import json
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from osu_fusion.data.descriptors import NUM_DESCRIPTORS
from osu_fusion.data.encode import SEQ_DIM
from osu_fusion.data.const import AUDIO_DIM, MAX_LENGTH_FRAMES


class MapData(NamedTuple):
    x: torch.Tensor
    a: torch.Tensor
    c: torch.Tensor
    descriptor_indices: torch.Tensor
    mapper_indices: torch.Tensor
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

            if "descriptor_indices" in map_data:
                descriptor_indices = torch.from_numpy(map_data["descriptor_indices"][:].astype(np.int64)).to(
                    self.device,
                )
            else:
                descriptor_indices = torch.tensor([], dtype=torch.int64, device=self.device)

            if "mapper_indices" in map_data:
                mapper_indices = torch.from_numpy(map_data["mapper_indices"][:].astype(np.int64)).to(self.device)
            else:
                mapper_indices = torch.tensor([], dtype=torch.int64, device=self.device)

        if load_audio:
            audio_file = map_file.parent.parent.parent / spec_path
            with h5py.File(audio_file, "r") as audio_data:
                a = self._to_tensor(audio_data["a"][:])
        else:
            a = torch.zeros((x.shape[0], AUDIO_DIM), dtype=torch.float32)

        if any(torch.isnan(t).any() for t in [x, a, c]):
            msg = f"Invalid values in map file {map_file}"
            raise ValueError(msg)

        return MapData(
            x=x,
            a=a,
            c=c,
            descriptor_indices=descriptor_indices,
            mapper_indices=mapper_indices,
            spec_path=spec_path,
        )


def filter_maps(maps: List[Path], max_length: int = 0) -> List[Path]:
    filtered = []
    for path in tqdm(maps, desc="Filtering dataset...", dynamic_ncols=True):
        try:
            with h5py.File(path, "r") as f:
                x_len = f["x"].shape[0]
                if x_len > MAX_LENGTH_FRAMES:
                    continue
                if max_length > 0 and x_len > max_length:
                    continue
                if f["x"].shape[1] != SEQ_DIM:
                    continue

                spec_path = f["spec_path"][()].decode("utf-8")
                audio_file = path.parent.parent.parent / spec_path
                if not audio_file.exists():
                    continue

                if "descriptor_indices" not in f:
                    continue

                if "mapper_indices" not in f:
                    continue
            filtered.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
    print(f"Filtered dataset: {len(filtered)}/{len(maps)} maps")
    return filtered


def count_num_mappers(dataset_path: Path) -> int:
    with open(dataset_path / "mapper_index.json", "r") as f:
        mapper_index = json.load(f)
    return max(int(v) for v in mapper_index.values()) + 1


class BeatmapDataset(Dataset):
    def __init__(self: "BeatmapDataset", **kwargs: dict) -> None:
        super().__init__()
        self.dataset = kwargs.pop("dataset")
        self.load_audio = kwargs.pop("load_audio", True)
        self.num_mappers: int = kwargs.pop("num_mappers", 0)

        self.tensor_loader = TensorLoader()

    @staticmethod
    def _indices_to_multihot(indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
        multihot = torch.zeros(vocab_size, dtype=torch.float32)
        for idx in indices.tolist():
            idx = int(idx)
            if 0 <= idx < vocab_size:
                multihot[idx] = 1.0
        return multihot

    def __len__(self: "BeatmapDataset") -> int:
        return len(self.dataset)

    def __getitem__(
        self: "BeatmapDataset",
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        map_data = self.tensor_loader.load_tensor(self.dataset[index], self.load_audio)
        descriptors = self._indices_to_multihot(map_data.descriptor_indices, NUM_DESCRIPTORS)
        mappers = self._indices_to_multihot(map_data.mapper_indices, self.num_mappers + 1)
        return map_data.x, map_data.a, map_data.c, descriptors, mappers
