import os
import time
import warnings
from multiprocessing import Lock
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
from audioread.ffdec import FFmpegAudioFile

from osu_fusion.library.osu.beatmap import Beatmap
from osu_fusion.library.osu.data.encode import encode_beatmap

# Constants
SR = 22050
MS_PER_FRAME = 8
HOP_LENGTH = (SR // 1000) * MS_PER_FRAME

FMIN = librosa.note_to_hz("C1")
N_OCTAVES = 7
OCTAVE_BINS = 12
AUDIO_DIM = N_OCTAVES * OCTAVE_BINS
CONTEXT_DIM = 5

# Supress FutureWarnings from librosa since we are using an older version
warnings.filterwarnings("ignore", category=FutureWarning)

if os.getenv("CREATE_DATASET"):
    from rosu_pp_py import Beatmap as RosuBeatmap
    from rosu_pp_py import Difficulty as RosuDifficulty

    _global_lock: Dict[str, Lock] = {}  # type: ignore

# Precompute the VQT parameters to avoid recalculating them
VQT_PARAMS = {
    "sr": SR,
    "hop_length": HOP_LENGTH,
    "fmin": FMIN,
    "n_bins": AUDIO_DIM,
    "bins_per_octave": OCTAVE_BINS,
}


def load_audio(audio_file: Path) -> np.ndarray:
    try:
        with FFmpegAudioFile(audio_file) as aro:
            wave, _ = librosa.load(aro, sr=SR, mono=True, dtype=np.float32)
    except Exception as e:
        msg = f"Error loading audio file {audio_file}: {e}"
        raise ValueError(msg) from e

    if wave.size == 0:
        msg = f"Empty audio file: {audio_file}"
        raise ValueError(msg)

    vqt = np.abs(librosa.vqt(y=wave, **VQT_PARAMS))
    return vqt.astype(np.float32)


def normalize_context(context: np.ndarray) -> np.ndarray:
    context[:4] = context[:4] / 5.0 - 1.0  # CS, AR, OD, HP
    context[4] = context[4] / 10.0 - 1.0  # SR
    return context


def unnormalize_context(context: np.ndarray) -> np.ndarray:
    context[:4] = (context[:4] + 1.0) * 5.0  # CS, AR, OD, HP
    context[4] = (context[4] + 1.0) * 10.0  # SR
    return context


def get_lock(path: Path) -> Lock:  # type: ignore
    path_str = str(path)
    if path_str not in _global_lock:
        _global_lock[path_str] = Lock()
    return _global_lock[path_str]


def get_audio_spec(beatmap: Beatmap, audio_file: Path) -> Optional[np.ndarray]:
    with get_lock(audio_file):
        if audio_file.exists():
            for attempt in range(5):  # Exponential backoff
                try:
                    with np.load(audio_file) as data:
                        spec = data["a"]
                    return spec
                except (ValueError, EOFError):
                    time.sleep(0.001 * (2**attempt))
                except Exception as e:
                    print(f"Unexpected error loading spec {audio_file}: {e}")
                    break
            # If all attempts fail, attempt to reload the audio
            audio_file.unlink(missing_ok=True)
            try:
                spec = load_audio(beatmap.audio_filename)
                np.savez_compressed(audio_file, a=spec)
                return spec
            except Exception as e:
                print(f"Failed to reload audio {beatmap.audio_filename}: {e}")
                return None
        else:
            try:
                spec = load_audio(beatmap.audio_filename)
                audio_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(audio_file, a=spec)
                return spec
            except Exception as e:
                print(f"Failed to load and save audio {beatmap.audio_filename}: {e}")
                return None


def prepare_map(data_dir: Path, map_file: Path) -> None:
    try:
        beatmap = Beatmap(map_file, meta_only=True)
    except Exception as e:
        print(f"[Error] Failed to load beatmap {map_file}: {e}")
        return

    if beatmap.mode != 0:
        return  # Only process standard mode beatmaps

    # Create a unique directory name based on the audio filename
    audio_stem = beatmap.audio_filename.stem
    audio_suffix = "".join(beatmap.audio_filename.suffixes)
    audio_file_dir = f"{audio_stem}_{audio_suffix.lstrip('.')}"
    map_dir = data_dir / map_file.parent.name / audio_file_dir

    spec_path = map_dir / "spec.npz"
    map_path = map_dir.parent / f"{map_file.stem}.map.npz"

    try:
        with open(map_file, "r", encoding="utf-8") as f:
            rosu_beatmap = RosuBeatmap(content=f.read())
        rosu_difficulty = RosuDifficulty()
        sr = rosu_difficulty.calculate(rosu_beatmap).stars
        sr = np.clip(sr, 0, 20)  # Clip SR to [0, 20]
        map_difficulty = np.array(
            [
                rosu_beatmap.cs,
                rosu_beatmap.ar,
                rosu_beatmap.od,
                rosu_beatmap.hp,
                sr,
            ],
            dtype=np.float32,
        )
    except Exception as e:
        print(f"[Error] Rosu failed to process beatmap {map_file}: {e}")
        return

    try:
        beatmap.parse_map_data()
    except Exception as e:
        print(f"[Error] Failed to parse beatmap data {map_file}: {e}")
        return

    spec = get_audio_spec(beatmap, spec_path)
    if spec is None:
        return

    frame_times = (
        librosa.frames_to_time(
            np.arange(spec.shape[-1]),
            sr=SR,
            hop_length=HOP_LENGTH,
        )
        * 1000
    )

    x = encode_beatmap(beatmap, frame_times)
    c = normalize_context(map_difficulty)

    # Save the processed map data
    try:
        spec_relative = spec_path.relative_to(map_path.parent).as_posix()
        np.savez_compressed(map_path, x=x, c=c, spec_path=spec_relative)
    except Exception as e:
        print(f"[Error] Failed to save map data {map_path}: {e}")
