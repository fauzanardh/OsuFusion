from typing import List, Tuple

import librosa


SR = 16000
MS_PER_FRAME = 16
HOP_LENGTH = (SR // 1000) * MS_PER_FRAME
MAX_LENGTH_FRAMES = 16384

FMIN = librosa.note_to_hz("C0")
N_OCTAVES = 8
OCTAVE_BINS = 12

# 96 VQT Bins + 1 Onset Envelope + 1 Metronome Phase = 98
AUDIO_DIM = (N_OCTAVES * OCTAVE_BINS) + 2
CONTEXT_DIM = 8

# classic ≤2012, transitional 2013-2016, modern 2017-2020, current 2021+
ERA_LABELS: List[str] = ["classic", "transitional", "modern", "current"]
ERA_BOUNDARIES: List[Tuple[str, int]] = [
    ("classic", 2012),
    ("transitional", 2016),
    ("modern", 2020),
]
NUM_ERAS: int = len(ERA_LABELS)


def year_to_era_index(year: int) -> int:
    for idx, (_, max_year) in enumerate(ERA_BOUNDARIES):
        if year <= max_year:
            return idx
    return NUM_ERAS - 1
