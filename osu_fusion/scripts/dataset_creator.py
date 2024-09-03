import os
import time
import warnings
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockBase
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import numpy.typing as npt
from audioread.ffdec import FFmpegAudioFile

from osu_fusion.library.osu.beatmap import Beatmap
from osu_fusion.library.osu.data.encode import encode_beatmap

SR = 22050
MS_PER_FRAME = 8
HOP_LENGTH = (SR // 1000) * MS_PER_FRAME

FMIN = librosa.note_to_hz("C0")
N_OCTAVES = 8
OCTAVE_BINS = 12
AUDIO_DIM = N_OCTAVES * OCTAVE_BINS
CONTEXT_DIM = 5

warnings.filterwarnings("ignore", category=FutureWarning)

if "CREATE_DATASET" in os.environ:
    from rosu_pp_py import Beatmap as RosuBeatmap
    from rosu_pp_py import Difficulty as RosuDifficulty

    _global_lock: Dict[str, LockBase] = {}


def load_audio(audio_file: Path) -> npt.NDArray:
    aro = FFmpegAudioFile(audio_file)
    wave, _ = librosa.load(aro, sr=SR, res_type="kaiser_best")
    if wave.shape[0] == 0:
        msg = f"Empty audio file: {audio_file}"
        raise ValueError(msg)

    return np.log(
        np.abs(
            librosa.vqt(
                y=wave,
                sr=SR,
                hop_length=HOP_LENGTH,
                fmin=FMIN,
                n_bins=AUDIO_DIM,
                bins_per_octave=OCTAVE_BINS,
            ),
        )
        + 1e-10,
    )


def normalize_context(context: npt.NDArray) -> npt.NDArray:
    # normalized CS, AR, OD, HP to [-1, 1] from [0, 10]
    context[0] = context[0] / 5 - 1
    context[1] = context[1] / 5 - 1
    context[2] = context[2] / 5 - 1
    context[3] = context[3] / 5 - 1
    # normalized SR to [-1, 1] from [0, 20]
    context[4] = context[4] / 10 - 1

    return context


def unnormalize_context(context: npt.NDArray) -> npt.NDArray:
    # unnormalized CS, AR, OD, HP from [-1, 1] to [0, 10]
    context[0] = (context[0] + 1) * 5
    context[1] = (context[1] + 1) * 5
    context[2] = (context[2] + 1) * 5
    context[3] = (context[3] + 1) * 5
    # unnormalized SR from [-1, 1] to [0, 20]
    context[4] = (context[4] + 1) * 10

    return context


def get_lock(path: Path) -> LockBase:
    return _global_lock.setdefault(str(path), Lock())


def get_audio_spec(beatmap: Beatmap, audio_file: Path) -> Optional[npt.NDArray]:
    with get_lock(audio_file):
        if audio_file.exists():
            for i in range(5):  # backoff
                try:
                    spec = np.load(audio_file)["a"]
                    return spec
                except ValueError:
                    time.sleep(0.001 * 2**i)
                except EOFError:
                    audio_file.unlink()
                    try:
                        spec = load_audio(beatmap.audio_filename)
                        return spec
                    except Exception as e:
                        print(f"Failed to load audio {beatmap.audio_filename}: {e}")
                        return
            else:
                print(f"Failed to load spec {audio_file}")
                return
        else:
            try:
                spec = load_audio(beatmap.audio_filename)
            except Exception as e:
                print(f"Failed to load audio {beatmap.audio_filename}: {e}")
                return

            audio_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(audio_file, a=spec)

            return spec


def prepare_map(data_dir: Path, map_file: Path) -> None:
    try:
        beatmap = Beatmap(map_file, meta_only=True)
    except Exception as e:
        print(f"Library failed to load beatmap {map_file}: {e}")
        return

    if beatmap.mode != 0:
        return

    audio_file_dir = "_".join([beatmap.audio_filename.stem, *(s[1:] for s in beatmap.audio_filename.suffixes)])
    map_dir = data_dir / map_file.parent.name / audio_file_dir

    spec_path = map_dir / "spec.npz"
    map_path = map_dir.parent / f"{map_file.stem}.map.npz"

    if spec_path.exists() and map_path.exists():
        return

    try:
        with open(map_file, "r", encoding="utf-8") as f:
            rosu_beatmap = RosuBeatmap(content=f.read())
        rosu_difficulty = RosuDifficulty()
        sr = rosu_difficulty.calculate(rosu_beatmap).stars
        # clip SR to [0, 20)
        sr = min(max(sr, 0), 20)
        map_difficulty = [
            rosu_beatmap.cs,
            rosu_beatmap.ar,
            rosu_beatmap.od,
            rosu_beatmap.hp,
            sr,
        ]
    except Exception as e:
        print(f"Rosu failed to load beatmap {map_file}: {e}")
        return

    try:
        beatmap.parse_map_data()
    except Exception as e:
        print(f"Library failed to parse beatmap {map_file}: {e}")
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
    c = normalize_context(np.array(map_difficulty, dtype=np.float32))

    # Get relative path for spec_path
    spec_path = spec_path.relative_to(map_path.parent)
    np.savez_compressed(map_path, x=x, c=c, spec_path=str(spec_path).replace("\\", "/"))
