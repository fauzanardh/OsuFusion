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
        # cursor signals
        "CURSOR_X",
        "CURSOR_Y",
    ],
    start=0,
)
