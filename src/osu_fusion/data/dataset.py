import random
from functools import partial
from pathlib import Path
from typing import Dict, Generator, NamedTuple, Optional, Tuple

import librosa
import numpy as np
import torch
from rosu_pp_py import Beatmap as RosuBeatmap
from rosu_pp_py import Difficulty as RosuDifficulty
from torch.utils.data import IterableDataset

from osu_fusion.data.augment import flip_cursor_horizontal, flip_cursor_vertical
from osu_fusion.data.const import (
    AUDIO_DIM,
    HOP_LENGTH,
    SR,
)
from osu_fusion.data.decode import Metadata, decode_beatmap
from osu_fusion.data.prepare_data import normalize_context, unnormalize_context


class ContextGenerator:
    def __init__(self: "ContextGenerator", sr: int = SR, hop_length: int = HOP_LENGTH) -> None:
        self.sr = sr
        self.hop_length = hop_length

        self.frames_to_ms = partial(
            librosa.frames_to_time,
            sr=sr,
            hop_length=hop_length,
        )

    @torch.no_grad()
    def get_new_context(self: "ContextGenerator", x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        cs, ar, od, hp, original_sr = unnormalize_context(c.clone()).tolist()

        n_frames = x.shape[-1]
        frames_indices = np.arange(n_frames)
        frame_times = self.frames_to_ms(frames_indices) * 1000

        x_numpy = x.cpu().numpy()
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

        try:
            segment_osu = decode_beatmap(metadata, x_numpy, frame_times, bpm=None, allow_beat_snap=False, verbose=False)
            segment_beatmap = RosuBeatmap(content=segment_osu)
            rosu_difficulty = RosuDifficulty()
            segment_sr = rosu_difficulty.calculate(segment_beatmap).stars
        except Exception as e:
            print(f"Error calculating SR: {e}")
            segment_sr = original_sr

        c = normalize_context(np.array([cs, ar, od, hp, segment_sr], dtype=np.float32))
        return torch.from_numpy(c)


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
        map_data = np.load(map_file)

        tensors = {
            "x": self._to_tensor(map_data["x"]),
            "c": self._to_tensor(map_data["c"]),
        }

        if load_audio:
            audio_file = map_file.parent.parent.parent / map_data["spec_path"].tolist()
            audio_data = np.load(audio_file)
            tensors["a"] = self._to_tensor(audio_data["a"])
        else:
            # Dummy audio tensor
            tensors["a"] = torch.zeros((AUDIO_DIM, tensors["x"].shape[-1]), dtype=torch.float32)

        if any(torch.isnan(t).any() for t in tensors.values()):
            msg = "Invalid values in map file"
            raise ValueError(msg)

        return MapData(
            x=tensors["x"],
            a=tensors["a"],
            c=tensors["c"],
            spec_path=map_data["spec_path"].tolist(),
        )


class StreamPerSample(IterableDataset):
    def __init__(self: "StreamPerSample", **kwargs: dict) -> None:
        super().__init__()
        self.dataset = kwargs.pop("dataset")
        self.sample_density = kwargs.pop("sample_density", 1.0)
        self.segment_sr = kwargs.pop("segment_sr", True)
        self.flip_horizontal_prob = kwargs.pop("flip_horizontal_prob", 0.5)
        self.flip_vertical_prob = kwargs.pop("flip_vertical_prob", 0.5)
        self.load_audio = kwargs.pop("load_audio", True)

        self.tensor_loader = TensorLoader()
        self.context_generator = ContextGenerator()

        if not (0 < self.sample_density <= 1):
            msg = "sample_density must be between 0 and 1"
            raise ValueError(msg)

    def sample_stream(
        self: "StreamPerSample",
        map_file: Path,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        raise NotImplementedError

    def __iter__(self: "StreamPerSample") -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
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
        num_samples = int(len(self.dataset) * self.sample_density)
        enumerated_dataset = list(enumerate(self.dataset))
        samples = random.sample(enumerated_dataset, num_samples)

        for i, sample in samples:
            if i % num_workers != worker_id:
                continue

            try:
                for x, a, c in self.sample_stream(sample):
                    yield x, a, c
            except Exception as e:
                print(f"Error processing sample {sample}: {e}")
                continue

        # Randomize the dataset order for each epoch
        random.shuffle(self.dataset)


class FullSequenceDataset(StreamPerSample):
    MAX_LENGTH = 65536

    def sample_stream(
        self: "FullSequenceDataset",
        map_file: Path,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        map_data = self.tensor_loader.load_tensor(map_file, self.load_audio)
        x, a, c = map_data.x, map_data.a, map_data.c

        if x.shape[-1] > self.MAX_LENGTH:
            return

        x = x[..., : self.MAX_LENGTH]
        a = a[..., : self.MAX_LENGTH]

        if random.random() < self.flip_horizontal_prob:
            x = flip_cursor_horizontal(x)
        if random.random() < self.flip_vertical_prob:
            x = flip_cursor_vertical(x)
        if self.segment_sr:
            c = self.context_generator.get_new_context(x, c)

        yield x, a, c


class SubsequenceDataset(StreamPerSample):
    def __init__(self: "SubsequenceDataset", **kwargs: Dict) -> None:
        super().__init__(**kwargs)
        self.sequence_length = kwargs.pop("sequence_length", 4096)

    def sample_stream(
        self: "SubsequenceDataset",
        map_file: Path,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        try:
            map_data = self.tensor_loader.load_tensor(map_file, self.load_audio)
            x, a, c = map_data.x, map_data.a, map_data.c
        except ValueError:
            return

        n = x.shape[-1]
        if self.sequence_length > n:
            return

        start = random.randint(0, n - self.sequence_length)
        x = x[..., start : start + self.sequence_length]
        a = a[..., start : start + self.sequence_length]

        if random.random() < self.flip_horizontal_prob:
            x = flip_cursor_horizontal(x)
        if random.random() < self.flip_vertical_prob:
            x = flip_cursor_vertical(x)
        if self.segment_sr:
            c = self.context_generator.get_new_context(x, c)

        yield x, a, c
