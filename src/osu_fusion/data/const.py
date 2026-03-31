import librosa


SR = 16000
MS_PER_FRAME = 32
HOP_LENGTH = (SR // 1000) * MS_PER_FRAME

FMIN = librosa.note_to_hz("C0")
N_OCTAVES = 8
OCTAVE_BINS = 12

# 96 VQT Bins + 1 Onset Envelope + 1 Metronome Phase = 98
AUDIO_DIM = (N_OCTAVES * OCTAVE_BINS) + 2
CONTEXT_DIM = 7
