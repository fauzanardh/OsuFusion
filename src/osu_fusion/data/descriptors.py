from typing import Dict, List

# omdb ID → full hierarchical tag path (content tags only)
OMDB_DESCRIPTORS: Dict[int, str] = {
    # gimmick
    12: "gimmick",
    69: "gimmick/slider_only",
    70: "gimmick/circle_only",
    # style
    46: "style",
    2: "style/messy",
    5: "style/geometric",
    7: "style/geometric/grid_snap",
    8: "style/geometric/hexgrid",
    6: "style/freeform",
    9: "style/symmetrical",
    49: "style/clean",
    50: "style/slidershapes",
    53: "style/distance_snapped",
    58: "style/inis_style",
    63: "style/avant_garde",
    76: "style/perfect_stacks",
    # expression
    47: "expression",
    3: "expression/simple",
    4: "expression/chaotic",
    13: "expression/repetition",
    14: "expression/progression",
    15: "expression/high_contrast",
    16: "expression/improvisation",
    60: "expression/playfield_usage",
    10: "expression/playfield_usage/playfield_constraint",
    62: "expression/difficulty_spike",
    66: "expression/low_sv",
    67: "expression/high_sv",
    # skillsets
    48: "skillset",
    17: "skillset/tech",
    18: "skillset/tech/slider_tech",
    20: "skillset/tech/complex_sv",
    32: "skillset/reading",
    33: "skillset/reading/visually_dense",
    34: "skillset/reading/overlap_reading",
    36: "skillset/alt",
    45: "skillset/aim",
    21: "skillset/aim/jump_aim",
    22: "skillset/aim/sharp_aim",
    23: "skillset/aim/wide_aim",
    24: "skillset/aim/linear_aim",
    25: "skillset/aim/aim_control",
    26: "skillset/aim/flow_aim",
    35: "skillset/aim/precision",
    74: "skillset/tap",
    19: "skillset/tap/finger_control",
    27: "skillset/tap/bursts",
    28: "skillset/tap/streams",
    29: "skillset/tap/streams/spaced_streams",
    31: "skillset/tap/streams/cutstreams",
    30: "skillset/tap/stamina",
}

# Build forward and reverse lookups
DESCRIPTOR_TAGS: List[str] = sorted(set(OMDB_DESCRIPTORS.values()))
DESCRIPTOR_NAME_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(DESCRIPTOR_TAGS)}
OMDB_ID_TO_IDX: Dict[int, int] = {omdb_id: DESCRIPTOR_NAME_TO_IDX[name] for omdb_id, name in OMDB_DESCRIPTORS.items()}
NUM_DESCRIPTORS: int = len(DESCRIPTOR_TAGS)

# Precomputed: for each tag index, the list of ancestor indices (including itself)
DESCRIPTOR_ANCESTORS: Dict[int, List[int]] = {}
for _tag_name, _tag_idx in DESCRIPTOR_NAME_TO_IDX.items():
    _parts = _tag_name.split("/")
    _ancestors = []
    for _i in range(len(_parts)):
        _prefix = "/".join(_parts[: _i + 1])
        if _prefix in DESCRIPTOR_NAME_TO_IDX:
            _ancestors.append(DESCRIPTOR_NAME_TO_IDX[_prefix])
    DESCRIPTOR_ANCESTORS[_tag_idx] = _ancestors

# Flat descriptor name (from old CSV) → hierarchical tag name
# Only includes content-detectable tags; unknown flat names return -1
FLAT_NAME_TO_DESCRIPTOR: Dict[str, str] = {
    "aim control": "skillset/aim/aim_control",
    "alt": "skillset/alt",
    "avant-garde": "style/avant_garde",
    "bursts": "skillset/tap/bursts",
    "chaotic": "expression/chaotic",
    "circle only": "gimmick/circle_only",
    "clean": "style/clean",
    "complex sv": "skillset/tech/complex_sv",
    "cutstreams": "skillset/tap/streams/cutstreams",
    "difficulty spike": "expression/difficulty_spike",
    "distance snapped": "style/distance_snapped",
    "finger control": "skillset/tap/finger_control",
    "flow aim": "skillset/aim/flow_aim",
    "freeform": "style/freeform",
    "geometric": "style/geometric",
    "gimmick": "gimmick",
    "grid snap": "style/geometric/grid_snap",
    "hexgrid": "style/geometric/hexgrid",
    "high contrast": "expression/high_contrast",
    "high sv": "expression/high_sv",
    "iNiS-style": "style/inis_style",
    "improvisation": "expression/improvisation",
    "jump aim": "skillset/aim/jump_aim",
    "linear aim": "skillset/aim/linear_aim",
    "low sv": "expression/low_sv",
    "messy": "style/messy",
    "overlap reading": "skillset/reading/overlap_reading",
    "perfect stacks": "style/perfect_stacks",
    "playfield constraint": "expression/playfield_usage/playfield_constraint",
    "playfield usage": "expression/playfield_usage",
    "precision": "skillset/aim/precision",
    "progression": "expression/progression",
    "reading": "skillset/reading",
    "repetition": "expression/repetition",
    "sharp aim": "skillset/aim/sharp_aim",
    "simple": "expression/simple",
    "slider only": "gimmick/slider_only",
    "slider tech": "skillset/tech/slider_tech",
    "slidershapes": "style/slidershapes",
    "spaced streams": "skillset/tap/streams/spaced_streams",
    "stamina": "skillset/tap/stamina",
    "streams": "skillset/tap/streams",
    "symmetrical": "style/symmetrical",
    "tech": "skillset/tech",
    "visually dense": "skillset/reading/visually_dense",
    "wide aim": "skillset/aim/wide_aim",
}


def flat_name_to_idx(flat_name: str) -> int:
    """Convert a flat descriptor name (from CSV) to an index. Returns -1 if not a content tag."""
    hierarchical = FLAT_NAME_TO_DESCRIPTOR.get(flat_name)
    if hierarchical is None:
        return -1
    return DESCRIPTOR_NAME_TO_IDX.get(hierarchical, -1)
