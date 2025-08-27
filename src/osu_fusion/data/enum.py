from enum import IntEnum

BeatmapEncoding = IntEnum(
    "BeatmapEncoding",
    [
        # Hit signals
        "HIT",
        "SUSTAIN",
        "SLIDER",
        "BEZIER_ANCHOR",
        "PERFECT_ANCHOR",
        "CATMULL_ANCHOR",
        "LINEAR_ANCHOR",
        "LAST_ANCHOR",
        "SLIDER_END",
        "SPINNER",
        "NEW_COMBO",
        "KIAI",
        # Cursor signals
        "CURSOR_X",
        "CURSOR_Y",
    ],
    start=0,
)
