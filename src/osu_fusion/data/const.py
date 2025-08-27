import librosa

from osu_fusion.data.enum import BeatmapEncoding

SR = 22050
MS_PER_FRAME = 8
HOP_LENGTH = (SR // 1000) * MS_PER_FRAME

FMIN = librosa.note_to_hz("C0")
N_OCTAVES = 8
OCTAVE_BINS = 12
AUDIO_DIM = N_OCTAVES * OCTAVE_BINS

HIT_DIM = len(BeatmapEncoding) - 2
CURSOR_DIM = 2
BEATMAP_DIM = HIT_DIM + CURSOR_DIM

CONTEXT_DIM = 7
