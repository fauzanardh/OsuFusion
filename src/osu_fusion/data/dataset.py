import json
import math
import random
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from osu_fusion.data.const import AUDIO_DIM, MAX_LENGTH_FRAMES
from osu_fusion.data.descriptors import NUM_DESCRIPTORS
from osu_fusion.data.encode import SEQ_DIM, SequenceEncoding

DEFAULT_BUCKET_BOUNDARIES = [1024, 2048, 4096, 8192, 16384]


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


def filter_maps(maps: List[Path], max_length: int = 0) -> Tuple[List[Path], List[int]]:
    filtered = []
    lengths = []
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
            lengths.append(x_len)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
    print(f"Filtered dataset: {len(filtered)}/{len(maps)} maps")
    return filtered, lengths


def build_metadata_cache(dataset_dir: Path, cache_path: Path) -> None:
    all_maps = list(dataset_dir.rglob("*.map.h5"))
    entries: list[dict[str, object]] = []
    skipped = 0

    for path in tqdm(all_maps, desc="Building metadata cache...", dynamic_ncols=True):
        try:
            with h5py.File(path, "r") as f:
                x_shape = list(f["x"].shape)
                spec_path = f["spec_path"][()].decode("utf-8")

                audio_file = path.parent.parent.parent / spec_path
                audio_exists = audio_file.exists()

                has_descriptors = "descriptor_indices" in f
                has_mappers = "mapper_indices" in f

                descriptor_indices = f["descriptor_indices"][:].astype(int).tolist() if has_descriptors else []

            entries.append(
                {
                    "path": str(path.relative_to(dataset_dir)),
                    "x_len": x_shape[0],
                    "x_dim": x_shape[1],
                    "audio_exists": audio_exists,
                    "has_descriptors": has_descriptors,
                    "has_mappers": has_mappers,
                    "descriptor_indices": descriptor_indices,
                },
            )
        except Exception as e:
            print(f"Skipping {path}: {e}")
            skipped += 1
            continue

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as fp:
        json.dump({"dataset_dir": str(dataset_dir), "entries": entries}, fp)

    print(f"Metadata cache written to {cache_path}")
    print(f"  Total entries: {len(entries)}, skipped: {skipped}")


def filter_maps_cached(
    cache_path: Path,
    dataset_dir: Path,
    max_length: int = 0,
) -> Tuple[List[Path], List[int], List[List[int]]]:
    with open(cache_path) as fp:
        cache = json.load(fp)

    filtered: List[Path] = []
    lengths: List[int] = []
    all_descriptor_indices: List[List[int]] = []

    for entry in cache["entries"]:
        x_len: int = entry["x_len"]
        x_dim: int = entry["x_dim"]
        audio_exists: bool = entry["audio_exists"]
        has_descriptors: bool = entry.get("has_descriptors", False)
        has_mappers: bool = entry.get("has_mappers", False)

        if x_len > MAX_LENGTH_FRAMES:
            continue
        if max_length > 0 and x_len > max_length:
            continue
        if x_dim != SEQ_DIM:
            continue
        if not audio_exists:
            continue
        if not has_descriptors:
            continue
        if not has_mappers:
            continue

        filtered.append(dataset_dir / entry["path"])
        lengths.append(x_len)
        all_descriptor_indices.append(entry.get("descriptor_indices", []))

    print(f"Filtered dataset (from cache): {len(filtered)}/{len(cache['entries'])} maps")
    return filtered, lengths, all_descriptor_indices


def count_num_mappers(dataset_path: Path) -> int:
    with open(dataset_path / "mapper_index.json", "r") as f:
        mapper_index = json.load(f)
    return max(int(v) for v in mapper_index.values()) + 1


def compute_sample_weights(
    all_descriptor_indices: List[List[int]],
) -> List[float]:
    idx_freq: Dict[int, int] = {}
    for desc_idx in all_descriptor_indices:
        for i in desc_idx:
            idx_freq[int(i)] = idx_freq.get(int(i), 0) + 1

    weights = []
    for desc_idx in all_descriptor_indices:
        if len(desc_idx) > 0:
            freqs = [idx_freq.get(int(i), 1) for i in desc_idx]
            min_freq = min(
                freqs,
            )  # Use minimum because we include ancestor descriptors too, which can be much more common
            weights.append(1.0 / math.sqrt(max(min_freq, 1.0)))
        else:
            weights.append(1.0)

    return weights


class BucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self: "BucketBatchSampler",
        lengths: List[int],
        batch_size: int,
        bucket_boundaries: Optional[List[int]] = None,
        drop_last: bool = False,
        seed: int = 0,
        sample_weights: Optional[List[float]] = None,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_boundaries = sorted(bucket_boundaries or DEFAULT_BUCKET_BOUNDARIES)
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.sample_weights = sample_weights

        self.buckets: List[List[int]] = [[] for _ in range(len(self.bucket_boundaries))]
        for idx, length in enumerate(lengths):
            bucket_id = self._get_bucket_id(length)
            self.buckets[bucket_id].append(idx)

        for i, bucket in enumerate(self.buckets):
            upper = self.bucket_boundaries[i]
            lower = self.bucket_boundaries[i - 1] + 1 if i > 0 else 1
            print(f"  Bucket {i} (len {lower}-{upper}): {len(bucket)} samples")

    def _get_bucket_id(self: "BucketBatchSampler", length: int) -> int:
        for i, boundary in enumerate(self.bucket_boundaries):
            if length <= boundary:
                return i
        return len(self.bucket_boundaries) - 1

    def set_epoch(self: "BucketBatchSampler", epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self: "BucketBatchSampler") -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        all_batches = []
        for bucket in self.buckets:
            if len(bucket) == 0:
                continue

            if self.sample_weights is not None:
                bucket_weights = [self.sample_weights[idx] for idx in bucket]
                indices = rng.choices(bucket, weights=bucket_weights, k=len(bucket))
            else:
                indices = bucket.copy()
                rng.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self: "BucketBatchSampler") -> int:
        count = 0
        for bucket in self.buckets:
            if self.drop_last:
                count += len(bucket) // self.batch_size
            else:
                count += math.ceil(len(bucket) / self.batch_size)
        return count


class BeatmapDataset(Dataset):
    def __init__(self: "BeatmapDataset", **kwargs: dict) -> None:
        super().__init__()
        self.dataset = kwargs.pop("dataset")
        self.lengths: List[int] = kwargs.pop("lengths", [])
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


def beatmap_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_lens = torch.tensor([x.shape[0] for x, _, _, _, _ in batch], dtype=torch.int32)
    max_len = max(x.shape[0] for x, _, _, _, _ in batch)

    padded_x = []
    padded_a = []
    for x, a, _, _, _ in batch:
        n_pad = max_len - x.shape[0]
        if n_pad > 0:
            x = F.pad(x, (0, 0, 0, n_pad), value=-1.0)
            a = F.pad(a, (0, 0, 0, n_pad))
        padded_x.append(x)
        padded_a.append(a)

    out_x = torch.stack(padded_x)
    out_a = torch.stack(padded_a)
    out_c = torch.stack([c for _, _, c, _, _ in batch])
    out_desc = torch.stack([d for _, _, _, d, _ in batch])
    out_mapper = torch.stack([m for _, _, _, _, m in batch])
    return out_x, out_a, out_c, out_desc, out_mapper, orig_lens


class ClassifierDataset(Dataset):
    def __init__(self: "ClassifierDataset", map_files: List[Path], augment: bool = True) -> None:
        super().__init__()
        self.map_files = map_files
        self.augment = augment

    def __len__(self: "ClassifierDataset") -> int:
        return len(self.map_files)

    @staticmethod
    def _augment(x: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            x[:, SequenceEncoding.X] = -x[:, SequenceEncoding.X]

        if random.random() < 0.5:
            x[:, SequenceEncoding.Y] = -x[:, SequenceEncoding.Y]

        if random.random() < 0.5:
            T = x.shape[0]
            crop_ratio = random.uniform(0.75, 1.0)
            crop_len = max(1, int(T * crop_ratio))
            start = random.randint(0, T - crop_len)
            x = x[start : start + crop_len]
            a = a[start : start + crop_len]

        return x, a

    def __getitem__(
        self: "ClassifierDataset",
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        map_file = self.map_files[index]
        with h5py.File(map_file, "r") as f:
            x = torch.from_numpy(f["x"][:]).float()
            c = torch.from_numpy(f["c"][:]).float()
            spec_path = f["spec_path"][()].decode("utf-8")

            # Load descriptor indices and convert to multi-hot
            if "descriptor_indices" in f:
                desc_idx = f["descriptor_indices"][:]
                descriptors = torch.zeros(NUM_DESCRIPTORS, dtype=torch.float32)
                for idx in desc_idx:
                    if 0 <= idx < NUM_DESCRIPTORS:
                        descriptors[idx] = 1.0
            else:
                descriptors = torch.zeros(NUM_DESCRIPTORS, dtype=torch.float32)

        audio_file = map_file.parent.parent.parent / spec_path
        with h5py.File(audio_file, "r") as audio_data:
            a = torch.from_numpy(audio_data["a"][:]).float()

        if torch.isnan(x).any() or torch.isnan(c).any() or torch.isnan(a).any():
            msg = f"NaN in {map_file}"
            raise ValueError(msg)

        if self.augment:
            x, a = self._augment(x, a)

        return x, a, c, descriptors


def classifier_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_lens = torch.tensor([x.shape[0] for x, _, _, _ in batch], dtype=torch.int32)
    max_len = max(x.shape[0] for x, _, _, _ in batch)

    padded_x = []
    padded_a = []
    for x, a, _, _ in batch:
        n_pad = max_len - x.shape[0]
        if n_pad > 0:
            x = F.pad(x, (0, 0, 0, n_pad))
            a = F.pad(a, (0, 0, 0, n_pad))
        padded_x.append(x)
        padded_a.append(a)

    out_x = torch.stack(padded_x)
    out_a = torch.stack(padded_a)
    out_c = torch.stack([c for _, _, c, _ in batch])
    out_tags = torch.stack([t for _, _, _, t in batch])
    return out_x, out_a, out_c, out_tags, orig_lens
