# ─────────────────────────────────────────────────────────────
#  Pitch Similarity Model — Similarity Engine
#
#  Z-score normalizes features within each pitch type, then
#  computes Euclidean distance between pitch profiles.
#  Comps are always within the same pitch type.
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import PITCH_FEATURES, TOP_N_COMPS

# Z-score column names
_Z_COLS = [f"z_{f}" for f in PITCH_FEATURES]

# Display columns included in comp output
_DISPLAY_COLS = [
    "pitcher_id", "pitcher_name", "throws", "year", "level", "pitch_type",
    "n_pitches", "usage_pct", "velo", "ivb", "hb", "extension",
    "release_height", "release_side", "spin_rate",
]


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalize each PITCH_FEATURES column within each pitch type.
    Adds z_{feature} columns to the DataFrame.
    Missing feature values are filled with the group mean before normalizing.
    """
    result = df.copy()

    for pt, grp in df.groupby("pitch_type"):
        idx = grp.index
        for col in PITCH_FEATURES:
            if col not in df.columns:
                result.loc[idx, f"z_{col}"] = 0.0
                continue

            vals   = grp[col].copy()
            # Fill NaN with group mean (handles sparse columns like spin_axis pre-2017)
            g_mean = vals.mean()
            vals   = vals.fillna(g_mean) if not np.isnan(g_mean) else vals.fillna(0.0)

            sigma = vals.std(ddof=0)
            if sigma > 0:
                result.loc[idx, f"z_{col}"] = (vals - vals.mean()) / sigma
            else:
                result.loc[idx, f"z_{col}"] = 0.0

    return result


def get_comps(
    pitcher_id: int,
    pitch_type: str,
    year: int,
    profiles_norm: pd.DataFrame,
    top_n: int = TOP_N_COMPS,
    same_hand_only: bool = False,
) -> pd.DataFrame:
    """
    Return the top-N most similar pitches to a given pitcher's
    pitch type in a given season.

    Comps are drawn from all seasons in profiles_norm (multi-year by default).
    The query pitcher's own row is excluded from results.

    Parameters
    ----------
    pitcher_id      : MLBAM pitcher ID
    pitch_type      : e.g. "FF", "SL", "CH"
    year            : season for the query pitcher
    profiles_norm   : DataFrame with z_{feature} columns from normalize_features()
    top_n           : number of comps to return
    same_hand_only  : if True, only return comps with the same throwing hand
    """
    z_cols = [c for c in _Z_COLS if c in profiles_norm.columns]

    # ── Query row ─────────────────────────────────────────────
    query_mask = (
        (profiles_norm["pitcher_id"] == pitcher_id) &
        (profiles_norm["pitch_type"] == pitch_type) &
        (profiles_norm["year"] == year)
    )
    query_rows = profiles_norm[query_mask]
    if query_rows.empty:
        return pd.DataFrame()

    query_vec = query_rows[z_cols].fillna(0).values[0]

    # ── Comp pool ─────────────────────────────────────────────
    pool = profiles_norm[
        (profiles_norm["pitch_type"] == pitch_type) &
        (profiles_norm["pitcher_id"] != pitcher_id)
    ].copy()

    if same_hand_only and "throws" in profiles_norm.columns:
        query_hand = query_rows["throws"].values[0]
        pool = pool[pool["throws"] == query_hand]

    if pool.empty:
        return pd.DataFrame()

    # ── Euclidean distance in z-score space ───────────────────
    pool_vecs  = pool[z_cols].fillna(0).values
    diffs      = pool_vecs - query_vec
    distances  = np.sqrt((diffs ** 2).sum(axis=1))

    pool = pool.copy()
    pool["distance"]   = distances
    pool["similarity"] = 1.0 / (1.0 + distances)   # 0–1, higher = more similar

    # ── Output columns ────────────────────────────────────────
    out_cols = [c for c in _DISPLAY_COLS if c in pool.columns]
    extra    = [c for c in ["era", "fip", "ip"] if c in pool.columns]
    out_cols = out_cols + extra + ["distance", "similarity"]

    return (
        pool.sort_values("distance")
            .head(top_n)[out_cols]
            .reset_index(drop=True)
    )


def get_arsenal_comps(
    pitcher_id: int,
    year: int,
    profiles_norm: pd.DataFrame,
    top_n: int = TOP_N_COMPS,
    same_hand_only: bool = False,
) -> dict:
    """
    Return comps for every pitch type in a pitcher's arsenal.

    Returns a dict keyed by pitch_type → comp DataFrame.
    """
    pitcher_rows = profiles_norm[
        (profiles_norm["pitcher_id"] == pitcher_id) &
        (profiles_norm["year"] == year)
    ]

    comps = {}
    for pt in pitcher_rows["pitch_type"].unique():
        comps[pt] = get_comps(
            pitcher_id, pt, year, profiles_norm,
            top_n=top_n, same_hand_only=same_hand_only,
        )
    return comps


def similarity_matrix(
    pitch_type: str,
    profiles_norm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a full pairwise similarity matrix for all profiles of
    a given pitch type. Returns a DataFrame indexed by pitcher_id+year.
    Useful for downstream ERA modelling.
    """
    z_cols = [c for c in _Z_COLS if c in profiles_norm.columns]
    pool = profiles_norm[profiles_norm["pitch_type"] == pitch_type].copy()

    if pool.empty:
        return pd.DataFrame()

    vecs = pool[z_cols].fillna(0).values
    labels = pool["pitcher_id"].astype(str) + "_" + pool["year"].astype(str)

    n = len(pool)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        diffs = vecs - vecs[i]
        dist_matrix[i] = np.sqrt((diffs ** 2).sum(axis=1))

    sim_matrix = 1.0 / (1.0 + dist_matrix)
    return pd.DataFrame(sim_matrix, index=labels.values, columns=labels.values)
