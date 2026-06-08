# ─────────────────────────────────────────────────────────────
#  Pitch Similarity Model — Predictive ERA
#
#  Predicts a pitcher's ERA by:
#    1. Finding the top-N most similar comps for each pitch type
#    2. Computing a similarity-weighted average ERA per pitch type
#    3. Blending pitch-type ERA predictions by usage %
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from similarity import get_comps
from config import TOP_N_COMPS


def predict_era(
    pitcher_id: int,
    year: int,
    profiles_norm: pd.DataFrame,
    top_n: int = TOP_N_COMPS,
    same_hand_only: bool = False,
) -> dict:
    """
    Predict ERA for a pitcher in a given season using pitch comps.

    Returns
    -------
    {
        "predicted_era":  float,   # usage-weighted blended ERA
        "n_comps":        int,     # total comp rows used
        "confidence":     float,   # fraction of arsenal with valid comps
        "pitch_details":  {
            pitch_type: {
                "predicted_era": float,
                "usage_pct":     float,
                "n_comps":       int,
                "top_comp":      str,   # name of closest comp
            }
        }
    }
    """
    pitcher_rows = profiles_norm[
        (profiles_norm["pitcher_id"] == pitcher_id) &
        (profiles_norm["year"] == year)
    ]

    if pitcher_rows.empty:
        return {"predicted_era": np.nan, "n_comps": 0, "confidence": 0.0, "pitch_details": {}}

    pitch_details  = {}
    weighted_sum   = 0.0
    usage_sum      = 0.0
    total_comps    = 0
    pitches_with_era = 0

    for _, row in pitcher_rows.iterrows():
        pt    = row["pitch_type"]
        usage = float(row["usage_pct"])

        comps = get_comps(
            pitcher_id, pt, year, profiles_norm,
            top_n=top_n, same_hand_only=same_hand_only,
        )

        if comps.empty or "era" not in comps.columns:
            pitch_details[pt] = {
                "predicted_era": np.nan,
                "usage_pct": usage,
                "n_comps": 0,
                "top_comp": None,
            }
            continue

        valid = comps.dropna(subset=["era"])
        if valid.empty:
            pitch_details[pt] = {
                "predicted_era": np.nan,
                "usage_pct": usage,
                "n_comps": 0,
                "top_comp": None,
            }
            continue

        # Similarity-weighted average ERA for this pitch type
        wts      = valid["similarity"].values
        era_vals = valid["era"].values
        pred     = float(np.average(era_vals, weights=wts))
        top_comp = valid.iloc[0]["pitcher_name"] if "pitcher_name" in valid.columns else None

        pitch_details[pt] = {
            "predicted_era": round(pred, 2),
            "usage_pct":     usage,
            "n_comps":       len(valid),
            "top_comp":      top_comp,
        }

        weighted_sum     += pred * usage
        usage_sum        += usage
        total_comps      += len(valid)
        pitches_with_era += 1

    overall    = round(weighted_sum / usage_sum, 2) if usage_sum > 0 else np.nan
    confidence = pitches_with_era / len(pitcher_rows) if len(pitcher_rows) > 0 else 0.0

    return {
        "predicted_era": overall,
        "n_comps":       total_comps,
        "confidence":    round(confidence, 2),
        "pitch_details": pitch_details,
    }


def batch_predict(
    profiles_norm: pd.DataFrame,
    year: int,
    top_n: int = TOP_N_COMPS,
    same_hand_only: bool = False,
) -> pd.DataFrame:
    """
    Run predict_era for every pitcher in a given season.
    Returns a DataFrame with pitcher_id, pitcher_name, year, predicted_era, confidence.
    """
    season_pitchers = profiles_norm[profiles_norm["year"] == year][
        ["pitcher_id", "pitcher_name"]
    ].drop_duplicates()

    rows = []
    total = len(season_pitchers)
    for i, (_, p) in enumerate(season_pitchers.iterrows(), 1):
        if i % 50 == 0 or i == total:
            print(f"  Predicting ERA: {i}/{total}", end="\r")
        result = predict_era(
            p["pitcher_id"], year, profiles_norm,
            top_n=top_n, same_hand_only=same_hand_only,
        )
        rows.append({
            "pitcher_id":    p["pitcher_id"],
            "pitcher_name":  p["pitcher_name"],
            "year":          year,
            "predicted_era": result["predicted_era"],
            "confidence":    result["confidence"],
            "n_comps":       result["n_comps"],
        })
    print()
    return pd.DataFrame(rows)
