import librosa

from osu_fusion.data.encode import SEQ_DIM

SR = 16000
MS_PER_FRAME = 32
HOP_LENGTH = (SR // 1000) * MS_PER_FRAME

FMIN = librosa.note_to_hz("C0")
N_OCTAVES = 8
OCTAVE_BINS = 12
AUDIO_DIM = N_OCTAVES * OCTAVE_BINS

BEATMAP_DIM = SEQ_DIM
CONTEXT_DIM = 7
