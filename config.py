# ─────────────────────────────────────────────────────────────
#  Pitch Similarity Model — Configuration
# ─────────────────────────────────────────────────────────────

# ── Season range ─────────────────────────────────────────────
STATCAST_FIRST_YEAR = 2015
STATCAST_LAST_YEAR  = 2026

# Baseball Savant started tracking Triple-A Statcast in 2023
MILB_FIRST_YEAR = 2023

# ── Similarity settings ──────────────────────────────────────
MIN_PITCHES  = 20    # minimum pitches thrown to include a pitch type
TOP_N_COMPS  = 10    # default number of comps to return

# ── Feature columns used for similarity ──────────────────────
# spin_axis is decomposed into sin/cos to handle circularity
PITCH_FEATURES = [
    "velo",
    "ivb",
    "hb",
    "extension",
    "release_height",
    "release_side",
    "spin_rate",
    "spin_axis_sin",
    "spin_axis_cos",
]

# ── Paths ────────────────────────────────────────────────────
DATA_DIR = "data"

# ── Pitch type display names ─────────────────────────────────
PITCH_TYPE_NAMES = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "SV": "Slurve",
    "CS": "Slow Curve",
    "FO": "Forkball",
    "KN": "Knuckleball",
    "EP": "Eephus",
    "SC": "Screwball",
}

# Pitch types to exclude from similarity (non-standard / noise)
EXCLUDE_PITCH_TYPES = {"PO", "IN", "AB", "FA", "UN", "XX"}
