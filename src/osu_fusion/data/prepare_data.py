import hashlib
from multiprocessing import Lock
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
from audioread.ffdec import FFmpegAudioFile
from rosu_pp_py import Beatmap as RosuBeatmap
from rosu_pp_py import Difficulty as RosuDifficulty

from osu_fusion.data.const import AUDIO_DIM, BEATMAP_DIM, CONTEXT_DIM, FMIN, HOP_LENGTH, OCTAVE_BINS, SR
from osu_fusion.data.encode import encode_beatmap
from osu_fusion.osu.beatmap import Beatmap

_global_lock: Dict[str, Lock] = {}  # type: ignore

VQT_PARAMS = {
    "sr": SR,
    "hop_length": HOP_LENGTH,
    "fmin": FMIN,
    "n_bins": AUDIO_DIM,
    "bins_per_octave": OCTAVE_BINS,
}


def compute_hash(audio_file: Path) -> str:
    hash_func = hashlib.sha256()
    try:
        with audio_file.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
    except Exception as e:
        print(f"[Error] Failed to compute hash for {audio_file}: {e}")
        return ""
    return hash_func.hexdigest()


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


def get_lock(path_str: str) -> Lock:  # type: ignore
    if path_str not in _global_lock:
        _global_lock[path_str] = Lock()
    return _global_lock[path_str]


def split_hash(hash_str: str) -> Tuple[str, str, str]:
    first_two = hash_str[:2]
    next_two = hash_str[2:4]
    remaining = hash_str[4:]
    return first_two, next_two, remaining


def get_audio_spec(beatmap: Beatmap, global_spec_dir: Path) -> Optional[Tuple[np.ndarray, str]]:
    audio_file = beatmap.audio_filename
    audio_hash = compute_hash(audio_file)
    if not audio_hash:
        return None

    # Split the hash to create hierarchical directories
    first_two, next_two, remaining_hash = split_hash(audio_hash)
    spec_filename = f"{remaining_hash}.spec.npz"
    spec_path = global_spec_dir / first_two / next_two / spec_filename

    lock = get_lock(str(spec_path))
    with lock:
        if spec_path.exists():
            try:
                with spec_path.open("rb") as f:
                    data = np.load(f)
                    spec = data["a"]
                return spec, audio_hash
            except (ValueError, EOFError):
                # Spec file is corrupted; attempt to regenerate
                spec_path.unlink(missing_ok=True)
                print(f"[Warning] Corrupted spec file {spec_path} removed.")
        # If spec does not exist or was corrupted, generate it
        try:
            spec = load_audio(audio_file)
            # Ensure the hierarchical spec directory exists
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            with spec_path.open("wb") as f:
                np.savez_compressed(f, a=spec)
            return spec, audio_hash
        except Exception as e:
            print(f"[Error] Failed to process audio {audio_file}: {e}")
            return None


def validate_map_data(map_file: Path, data_dir: Path) -> bool:
    try:
        with map_file.open("rb") as f:
            data = np.load(f)
            if "x" not in data or "c" not in data:
                print(f"[Error] Missing data in map file {map_file}")
                return False

            x = data["x"]
            c = data["c"]
            if x.shape[0] != BEATMAP_DIM or c.shape[0] != CONTEXT_DIM:
                print(f"[Error] Invalid data shape in map file {map_file}")
                return False

            if x.size == 0:
                print(f"[Error] Empty data in map file {map_file}")
                return False

            if "spec_path" not in data:
                print(f"[Error] Missing `spec_path` key in map file {map_file}")
                return False

            spec_relative = data["spec_path"].item()
            spec_file = data_dir / spec_relative
            if not spec_file.exists():
                print(f"[Error] Missing spec file {spec_file}")
                return False
    except Exception as e:
        print(f"[Error] Failed to load map data {map_file}: {e}")
        return False

    return True


def prepare_map(data_dir: Path, map_file: Path) -> None:
    try:
        beatmap = Beatmap(map_file, meta_only=True)
    except Exception as e:
        print(f"[Error] Failed to load beatmap {map_file}: {e}")
        return

    if beatmap.mode != 0:
        return  # Only process standard mode beatmaps

    # Define the global specs directory
    global_spec_dir = data_dir / "specs"

    # Define the map data path (unique per map)
    map_data_dir = data_dir / "maps" / map_file.parent.name
    map_data_dir.mkdir(parents=True, exist_ok=True)
    map_path = map_data_dir / f"{map_file.stem}.map.npz"

    # If the map data already exists, check if the map file is valid, then skip
    if map_path.exists() and validate_map_data(map_path, data_dir):
        return

    try:
        with map_file.open("r", encoding="utf-8") as f:
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

    spec_result = get_audio_spec(beatmap, global_spec_dir)
    if spec_result is None:
        return
    spec, audio_hash = spec_result

    frame_times = (
        librosa.frames_to_time(
            np.arange(spec.shape[-1]),
            sr=SR,
            hop_length=HOP_LENGTH,
        )
        * 1000
    )  # Convert to milliseconds

    x = encode_beatmap(beatmap, frame_times)
    c = normalize_context(map_difficulty)

    # Save the processed map data
    try:
        # Split the hash to reconstruct the relative path
        first_two, next_two, remaining_hash = split_hash(audio_hash)
        spec_relative = f"specs/{first_two}/{next_two}/{remaining_hash}.spec.npz"  # Store relative path to global specs
        with map_path.open("wb") as f:
            np.savez_compressed(f, x=x, c=c, spec_path=spec_relative)
    except Exception as e:
        print(f"[Error] Failed to save map data {map_path}: {e}")
