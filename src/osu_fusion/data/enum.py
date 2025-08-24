from enum import IntEnum

BeatmapEncoding = IntEnum(
    "BeatmapEncoding",
    [
        # hit signals
        "HIT",
        "SUSTAIN",
        "SLIDER",
        "WHITE_BEZIER_ANCHORS",
        "RED_BEZIER_ANCHORS",
        "LINE_ANCHORS",
        "PERFECT_ANCHORS",
        "COMBO",
        "KIAI",  # Only used to as an auxiliary signal so the model can learn better
        # cursor signals
        "CURSOR_X",
        "CURSOR_Y",
    ],
    start=0,
)
