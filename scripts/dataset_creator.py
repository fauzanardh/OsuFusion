import time
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt
from audioread.ffdec import FFmpegAudioFile
from rosu_pp_py import Beatmap as RosuBeatmap
from rosu_pp_py import Calculator as RosuCalculator

from library.osu.beatmap import Beatmap
from library.osu.from_beatmap import AUDIO_DIM, from_beatmap

N_FFT = 2048
N_MELS = 64
SR = 22050
FRAME_RATE = 6
HOP_LENGTH = int(SR * FRAME_RATE / 1000)

_calculator = RosuCalculator(mode=0)


def load_audio(audio_file: Path) -> npt.NDArray:
    aro = FFmpegAudioFile(audio_file)
    wave, _ = librosa.load(aro, sr=SR, res_type="kaiser_best")
    if wave.shape[0] == 0:
        msg = f"Empty audio file: {audio_file}"
        raise ValueError(msg)

    spec = librosa.feature.mfcc(
        y=wave,
        sr=SR,
        n_mfcc=AUDIO_DIM,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    return spec


def normalize_context(context: npt.NDArray) -> npt.NDArray:
    # normalized HP, CS, OD, AR to [-1, 1] from [0, 10]
    context[0] = context[0] / 5 - 1
    context[1] = context[1] / 5 - 1
    context[2] = context[2] / 5 - 1
    context[3] = context[3] / 5 - 1
    # normalized SR to [-1, 1] from [0, 20]
    context[4] = context[4] / 10 - 1
    # normalized BPM to [-1, 1] from [0, 300]
    context[5] = context[5] / 150 - 1

    return context


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
        beatmap_attributes = _calculator.map_attributes(rosu_beatmap)
        sr = _calculator.difficulty(rosu_beatmap).stars
        # clip SR to [0, 20)
        sr = min(max(sr, 0), 20)
        # clip BPM to [0, 300)
        bpm = min(max(beatmap_attributes.bpm, 0), 300)
        map_difficulty = [
            beatmap_attributes.hp,
            beatmap_attributes.cs,
            beatmap_attributes.od,
            beatmap_attributes.ar,
            sr,
            bpm,
        ]
    except Exception as e:
        print(f"Rosu failed to load beatmap {map_file}: {e}")
        return

    try:
        beatmap.parse_map_data()
    except Exception as e:
        print(f"Library failed to parse beatmap {map_file}: {e}")
        return

    if spec_path.exists():
        for i in range(5):  # backoff
            try:
                spec = np.load(spec_path)["a"]
                break
            except ValueError:
                time.sleep(0.001 * 2**i)
        else:
            print(f"Failed to load spec {spec_path}")
            return
    else:
        try:
            spec = load_audio(beatmap.audio_filename)
        except Exception as e:
            print(f"Failed to load audio {beatmap.audio_filename}: {e}")
            return

        spec_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(spec_path, a=spec)

    frame_times = (
        librosa.frames_to_time(
            np.arange(spec.shape[-1]),
            sr=SR,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
        )
        * 1000
    )

    x = from_beatmap(beatmap, frame_times)
    c = np.array(map_difficulty, dtype=np.float32)
    c = normalize_context(c)

    np.savez_compressed(map_path, x=x, c=c)
