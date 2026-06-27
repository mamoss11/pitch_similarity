# ─────────────────────────────────────────────────────────────
#  Pitch Similarity Model — Streamlit App
#
#  Run: python -m streamlit run app.py
# ─────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    STATCAST_FIRST_YEAR, STATCAST_LAST_YEAR, MILB_FIRST_YEAR,
    PITCH_TYPE_NAMES, TOP_N_COMPS,
)
from data_fetcher import (
    load_all_profiles, load_all_era, merge_era,
    fetch_season_profiles, fetch_era_data,
    load_milb_profiles, fetch_milb_season_profiles,
)
from similarity import normalize_features, get_comps
from era_model import predict_era, batch_predict

st.set_page_config(
    page_title="Pitch Similarity Model",
    page_icon="⚾",
    layout="wide",
)

# ── Data loading (cached) ─────────────────────────────────────

@st.cache_data(show_spinner="Loading pitch profiles...", ttl=24 * 3600)
def _load_profiles() -> pd.DataFrame:
    seasons  = list(range(STATCAST_FIRST_YEAR, STATCAST_LAST_YEAR + 1))
    profiles = load_all_profiles(seasons, cached_only=True)
    if profiles.empty:
        return profiles
    cached_years = sorted(profiles["year"].unique())
    era_all  = load_all_era(cached_years)
    profiles = merge_era(profiles, era_all)
    return profiles


@st.cache_data(show_spinner="Normalizing features...")
def _normalize(profiles_hash: int, _profiles: pd.DataFrame) -> pd.DataFrame:
    return normalize_features(_profiles)


@st.cache_data(show_spinner="Computing leaderboard...")
def _batch_predict(profiles_hash: int, _profiles: pd.DataFrame, year: int, top_n: int, same_hand: bool) -> pd.DataFrame:
    return batch_predict(_profiles, year, top_n=top_n, same_hand_only=same_hand)


@st.cache_data(show_spinner="Loading AAA profiles...", ttl=24 * 3600)
def _load_milb_profiles() -> pd.DataFrame:
    seasons = list(range(MILB_FIRST_YEAR, STATCAST_LAST_YEAR + 1))
    return load_milb_profiles(seasons, cached_only=True)


# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.title("⚾ Pitch Similarity")

    year = st.selectbox(
        "Season",
        options=list(range(STATCAST_LAST_YEAR, STATCAST_FIRST_YEAR - 1, -1)),
        index=0,
    )
    league = st.radio(
        "League",
        options=["MLB", "AAA", "Both"],
        index=0,
        help="AAA data available from 2023 onward. Comps always span both leagues.",
    )
    top_n = st.slider("Top N comps", min_value=3, max_value=25, value=TOP_N_COMPS)
    same_hand = st.checkbox("Same handedness only", value=False)

    st.divider()

    curr_year = date.today().year
    if st.button(f"Refresh {curr_year} Data", help="Re-fetch current season profiles and ERA from Statcast/FanGraphs"):
        with st.spinner(f"Fetching {curr_year} data (this takes a few minutes)..."):
            fetch_season_profiles(curr_year, force=True)
            fetch_era_data(curr_year, force=True)
            if curr_year >= MILB_FIRST_YEAR:
                fetch_milb_season_profiles(curr_year, force=True)
            _load_profiles.clear()
            _load_milb_profiles.clear()
            _normalize.clear()
            _batch_predict.clear()
        st.success(f"{curr_year} data refreshed.")
        st.rerun()

    st.divider()
    st.caption(
        "Data: Baseball Savant / Statcast via pybaseball.  "
        "ERA: FanGraphs.  "
        f"Seasons: {STATCAST_FIRST_YEAR}–{STATCAST_LAST_YEAR}."
    )

# ── Load data ─────────────────────────────────────────────────

try:
    profiles_raw = _load_profiles()
except Exception as e:
    st.error(f"Could not load profiles: {e}")
    st.info("Run `python main.py --fetch-all` first to build the data cache.")
    st.stop()

if profiles_raw.empty:
    st.warning("No profiles loaded. Run `python main.py --fetch-all` to build the cache.")
    st.stop()

# Load AAA profiles when requested
include_aaa = (league in ("AAA", "Both")) and (year >= MILB_FIRST_YEAR)
milb_raw = _load_milb_profiles() if include_aaa else pd.DataFrame()

if include_aaa and not milb_raw.empty:
    combined_raw = pd.concat([profiles_raw, milb_raw], ignore_index=True)
else:
    combined_raw = profiles_raw

profiles_norm = _normalize(id(profiles_raw) + (id(milb_raw) if include_aaa else 0), combined_raw)

# ── Pitcher search ────────────────────────────────────────────

st.title("Pitch Similarity Model")

tab_lookup, tab_board = st.tabs(["🔍 Pitcher Lookup", "🏆 Leaderboard"])

# ── Pitcher Lookup tab ────────────────────────────────────────

with tab_lookup:
    # Filter by selected league for the search dropdown
    season_profiles = profiles_norm[profiles_norm["year"] == year]
    if league == "MLB":
        season_profiles = season_profiles[season_profiles.get("level", "MLB") == "MLB"] if "level" in season_profiles.columns else season_profiles
    elif league == "AAA":
        if year < MILB_FIRST_YEAR:
            st.warning(f"AAA Statcast data is only available from {MILB_FIRST_YEAR} onward.")
            st.stop()
        if milb_raw.empty:
            st.warning(f"No AAA data cached for {year}. Click 'Refresh {year} Data' in the sidebar.")
            st.stop()
        season_profiles = season_profiles[season_profiles["level"] == "AAA"]

    if season_profiles.empty:
        st.warning(f"No data for {year}. Run `python main.py --fetch {year}` to fetch this season.")
    else:
        pitcher_options = (
            season_profiles[["pitcher_id", "pitcher_name"]]
            .drop_duplicates()
            .sort_values("pitcher_name")
        )

        search = st.text_input("Search pitcher", placeholder="e.g. Gerrit Cole")

        if not search:
            st.info("Type a pitcher name above to get started.")
        else:
            matches = pitcher_options[
                pitcher_options["pitcher_name"].str.contains(search, case=False, na=False)
            ]
            if matches.empty:
                st.warning(f"No pitcher matching '{search}' found in {year}.")
            else:
                if len(matches) == 1:
                    selected_name = matches["pitcher_name"].iloc[0]
                else:
                    selected_name = st.selectbox("Select pitcher", options=matches["pitcher_name"].tolist())

                pitcher_id   = pitcher_options[pitcher_options["pitcher_name"] == selected_name]["pitcher_id"].iloc[0]
                pitcher_rows = season_profiles[season_profiles["pitcher_id"] == pitcher_id]

                # ── Header ────────────────────────────────────────────
                col_name, col_era = st.columns([3, 1])
                with col_name:
                    throws = pitcher_rows["throws"].iloc[0] if "throws" in pitcher_rows.columns else "?"
                    st.header(f"{selected_name}  ({throws}HP, {year})")

                # ── Predicted ERA ──────────────────────────────────────
                era_result = predict_era(pitcher_id, year, profiles_norm, top_n=top_n, same_hand_only=same_hand)
                with col_era:
                    if not np.isnan(era_result["predicted_era"]):
                        st.metric(
                            label="Predicted ERA",
                            value=f"{era_result['predicted_era']:.2f}",
                            help=f"Comp-weighted ERA across arsenal. "
                                 f"Confidence: {era_result['confidence']:.0%} | {era_result['n_comps']} comps",
                        )
                        if "era" in pitcher_rows.columns:
                            actual = pitcher_rows["era"].dropna()
                            if not actual.empty:
                                st.metric(label="Actual ERA", value=f"{actual.iloc[0]:.2f}")

                st.divider()

                # ── Arsenal table ──────────────────────────────────────
                st.subheader("Arsenal")
                display_cols = ["pitch_type", "n_pitches", "usage_pct", "velo", "ivb", "hb",
                                "extension", "release_height", "release_side", "spin_rate"]
                display_cols = [c for c in display_cols if c in pitcher_rows.columns]

                arsenal_df = pitcher_rows[display_cols].copy()
                arsenal_df["pitch_name"] = arsenal_df["pitch_type"].map(lambda x: PITCH_TYPE_NAMES.get(x, x))
                arsenal_df = arsenal_df.sort_values("usage_pct", ascending=False)

                arsenal_display = arsenal_df[[
                    "pitch_name", "pitch_type", "n_pitches", "usage_pct",
                    "velo", "ivb", "hb", "extension", "release_height", "release_side", "spin_rate",
                ]].rename(columns={
                    "pitch_name":     "Pitch",
                    "pitch_type":     "Code",
                    "n_pitches":      "# Pitches",
                    "usage_pct":      "Usage %",
                    "velo":           "Velo",
                    "ivb":            "IVB (in)",
                    "hb":             "HB (in)",
                    "extension":      "Ext (ft)",
                    "release_height": "Rel Ht (ft)",
                    "release_side":   "Rel Side (ft)",
                    "spin_rate":      "Spin Rate",
                })

                st.dataframe(
                    arsenal_display.style.format({
                        "Usage %":       "{:.1%}",
                        "Velo":          "{:.1f}",
                        "IVB (in)":      "{:.1f}",
                        "HB (in)":       "{:.1f}",
                        "Ext (ft)":      "{:.2f}",
                        "Rel Ht (ft)":   "{:.2f}",
                        "Rel Side (ft)": "{:.2f}",
                        "Spin Rate":     "{:.0f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ── Comps per pitch type ───────────────────────────────
                st.subheader("Pitch Comps")
                pitch_tabs = list(arsenal_df.sort_values("usage_pct", ascending=False)["pitch_type"])
                tab_labels = [f"{PITCH_TYPE_NAMES.get(pt, pt)} ({pt})" for pt in pitch_tabs]

                if pitch_tabs:
                    tabs = st.tabs(tab_labels)
                    for tab, pt in zip(tabs, pitch_tabs):
                        with tab:
                            comp_df = get_comps(
                                pitcher_id, pt, year, profiles_norm,
                                top_n=top_n, same_hand_only=same_hand,
                            )
                            if comp_df.empty:
                                st.info("No comps found for this pitch type.")
                                continue

                            comp_display = comp_df.copy()
                            comp_display["pitch_name"] = comp_display["pitch_type"].map(
                                lambda x: PITCH_TYPE_NAMES.get(x, x)
                            )

                            show_cols = ["pitcher_name", "year", "level", "similarity", "velo", "ivb", "hb",
                                         "extension", "release_height", "release_side", "spin_rate"]
                            if "era" in comp_display.columns:
                                show_cols.append("era")
                            show_cols = [c for c in show_cols if c in comp_display.columns]

                            comp_display = comp_display[show_cols].rename(columns={
                                "pitcher_name":   "Pitcher",
                                "year":           "Year",
                                "level":          "Lg",
                                "similarity":     "Similarity",
                                "velo":           "Velo",
                                "ivb":            "IVB (in)",
                                "hb":             "HB (in)",
                                "extension":      "Ext (ft)",
                                "release_height": "Rel Ht (ft)",
                                "release_side":   "Rel Side (ft)",
                                "spin_rate":      "Spin Rate",
                                "era":            "ERA",
                            })

                            fmt = {
                                "Similarity":    "{:.1%}",
                                "Velo":          "{:.1f}",
                                "IVB (in)":      "{:.1f}",
                                "HB (in)":       "{:.1f}",
                                "Ext (ft)":      "{:.2f}",
                                "Rel Ht (ft)":   "{:.2f}",
                                "Rel Side (ft)": "{:.2f}",
                                "Spin Rate":     "{:.0f}",
                            }
                            if "ERA" in comp_display.columns:
                                fmt["ERA"] = "{:.2f}"

                            st.dataframe(
                                comp_display.style.format(fmt, na_rep="—"),
                                use_container_width=True,
                                hide_index=True,
                            )

                            detail = era_result.get("pitch_details", {}).get(pt, {})
                            if detail.get("predicted_era") and not np.isnan(detail["predicted_era"]):
                                st.caption(
                                    f"Predicted ERA for {PITCH_TYPE_NAMES.get(pt, pt)}: "
                                    f"**{detail['predicted_era']:.2f}** "
                                    f"({detail['n_comps']} comps, "
                                    f"closest: {detail['top_comp'] or 'N/A'})"
                                )

# ── Leaderboard tab ───────────────────────────────────────────

with tab_board:
    st.subheader(f"{year} Predicted ERA Leaderboard (MLB)")

    # Leaderboard always uses MLB-only profiles for ERA prediction
    mlb_norm = profiles_norm[profiles_norm["level"] == "MLB"] if "level" in profiles_norm.columns else profiles_norm
    lb = _batch_predict(id(mlb_norm), mlb_norm, year, top_n, same_hand)

    if lb.empty:
        st.warning("No leaderboard data available.")
    else:
        # Join actual ERA
        if "era" in mlb_norm.columns:
            actual_era = (
                mlb_norm[mlb_norm["year"] == year][["pitcher_id", "era"]]
                .dropna(subset=["era"])
                .drop_duplicates(subset=["pitcher_id"])
            )
            lb = lb.merge(actual_era, on="pitcher_id", how="left")
        else:
            lb["era"] = np.nan

        lb["diff"] = (lb["era"] - lb["predicted_era"]).round(2)
        lb = lb.dropna(subset=["predicted_era"]).sort_values("predicted_era").reset_index(drop=True)
        lb.insert(0, "#", range(1, len(lb) + 1))

        display = lb[["#", "pitcher_name", "predicted_era", "era", "diff", "confidence"]].rename(columns={
            "pitcher_name":  "Pitcher",
            "predicted_era": "Pred ERA",
            "era":           "Actual ERA",
            "diff":          "Diff (Act−Pred)",
            "confidence":    "Confidence",
        })

        st.dataframe(
            display.style.format({
                "Pred ERA":        "{:.2f}",
                "Actual ERA":      "{:.2f}",
                "Diff (Act−Pred)": "{:.2f}",
                "Confidence":      "{:.0%}",
            }, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )
