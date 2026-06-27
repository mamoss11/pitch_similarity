# ─────────────────────────────────────────────────────────────
#  Pitch Similarity Model — Data Fetcher
#
#  Pulls Statcast pitch-level data by season (month by month),
#  aggregates to pitcher × pitch_type profiles, and caches to CSV.
#  ERA data pulled separately from FanGraphs via pybaseball.
# ─────────────────────────────────────────────────────────────
import os
import time
import warnings
from datetime import date
import numpy as np
import pandas as pd
from pybaseball import statcast, pitching_stats_bref, cache

cache.enable()
warnings.filterwarnings("ignore")

from config import (
    STATCAST_FIRST_YEAR, STATCAST_LAST_YEAR,
    MIN_PITCHES, DATA_DIR, PITCH_FEATURES, EXCLUDE_PITCH_TYPES,
)

# Raw Statcast columns we care about
_STATCAST_COLS = [
    "pitcher", "player_name", "p_throws", "pitch_type",
    "release_speed", "pfx_x", "pfx_z",
    "release_extension", "release_pos_x", "release_pos_z",
    "release_spin_rate", "spin_axis",
]


# ── Season date ranges ────────────────────────────────────────

def _months_for_year(year: int) -> list:
    """Return (start, end) date strings covering the full regular season.

    For the current year, drops month windows that haven't started yet so we
    don't make empty API calls for future dates.
    """
    months = [
        (f"{year}-03-20", f"{year}-03-31"),  # opening week (some years)
        (f"{year}-04-01", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-10-05"),  # early October games
    ]
    if year == date.today().year:
        today_str = date.today().strftime("%Y-%m-%d")
        months = [(s, e) for s, e in months if s <= today_str]
    return months


# ── Raw → aggregated profiles ─────────────────────────────────

def _aggregate_raw(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Convert raw Statcast pitch rows into pitcher × pitch_type profile rows.

    One output row per (pitcher_id, pitch_type, year).
    Features are mean values across all pitches of that type.
    """
    # Keep only columns that exist
    keep = [c for c in _STATCAST_COLS if c in raw.columns]
    df = raw[keep].copy()

    # Drop missing / excluded pitch types
    df = df[df["pitch_type"].notna()]
    df = df[~df["pitch_type"].isin(EXCLUDE_PITCH_TYPES)]
    df = df[df["pitch_type"] != ""]

    # Drop rows missing core measurement columns
    df = df.dropna(subset=["release_speed", "pfx_x", "pfx_z"])

    if df.empty:
        return pd.DataFrame()

    # ── Normalise handedness ──────────────────────────────────
    # Flip horizontal features for LHP so arm-side is always positive.
    # Statcast pfx_x: positive = toward first base (catcher perspective).
    # RHP arm-side = first-base side → positive already correct.
    # LHP arm-side = third-base side → flip sign.
    lhp = df["p_throws"] == "L"
    df.loc[lhp, "pfx_x"]       = -df.loc[lhp, "pfx_x"]
    if "release_pos_x" in df.columns:
        df.loc[lhp, "release_pos_x"] = -df.loc[lhp, "release_pos_x"]

    # ── Derived features ──────────────────────────────────────
    df["velo"]           = df["release_speed"]
    df["ivb"]            = df["pfx_z"] * 12          # feet → inches
    df["hb"]             = df["pfx_x"] * 12           # feet → inches
    df["extension"]      = df.get("release_extension", np.nan)
    df["release_height"] = df.get("release_pos_z",     np.nan)
    df["release_side"]   = df.get("release_pos_x",     np.nan)
    df["spin_rate"]      = df.get("release_spin_rate", np.nan)

    # Spin axis → sin/cos (handles circular variable correctly)
    if "spin_axis" in df.columns:
        rad = np.radians(df["spin_axis"].astype(float))
        df["spin_axis_sin"] = np.sin(rad)
        df["spin_axis_cos"] = np.cos(rad)
    else:
        df["spin_axis_sin"] = np.nan
        df["spin_axis_cos"] = np.nan

    feat_cols = [
        "velo", "ivb", "hb", "extension",
        "release_height", "release_side",
        "spin_rate", "spin_axis_sin", "spin_axis_cos",
    ]

    # ── Aggregate ─────────────────────────────────────────────
    grp    = df.groupby(["pitcher", "pitch_type"])
    counts = grp.size().reset_index(name="n_pitches")
    means  = grp[feat_cols].mean().reset_index()
    meta   = grp[["player_name", "p_throws"]].first().reset_index()

    result = counts.merge(means, on=["pitcher", "pitch_type"])
    result = result.merge(meta,  on=["pitcher", "pitch_type"])

    # Apply minimum pitch threshold
    result = result[result["n_pitches"] >= MIN_PITCHES].copy()

    if result.empty:
        return pd.DataFrame()

    # Usage % (this pitch type / all pitches for this pitcher)
    total = result.groupby("pitcher")["n_pitches"].transform("sum")
    result["usage_pct"] = result["n_pitches"] / total

    result["year"] = year
    result = result.rename(columns={
        "pitcher":   "pitcher_id",
        "p_throws":  "throws",
    })

    # Convert Statcast "Last, First" name to "First Last" for display
    result["pitcher_name"] = result["player_name"].apply(_fmt_name)
    result = result.drop(columns=["player_name"])

    return result.reset_index(drop=True)


def _fmt_name(name: str) -> str:
    """Convert 'Last, First' → 'First Last'. Pass through if already formatted."""
    if not isinstance(name, str):
        return name
    parts = name.split(", ")
    return f"{parts[1]} {parts[0]}" if len(parts) == 2 else name


# ── Season fetch / cache ──────────────────────────────────────

_CACHE_MAX_AGE_SECONDS = 24 * 3600  # auto-refresh current season after 24 h


def _cache_is_stale(path: str) -> bool:
    """Return True if the file is older than _CACHE_MAX_AGE_SECONDS."""
    return (time.time() - os.path.getmtime(path)) > _CACHE_MAX_AGE_SECONDS


def fetch_season_profiles(year: int, force: bool = False) -> pd.DataFrame:
    """
    Load pitcher×pitch_type profiles for a season.
    Pulls from Statcast and caches to data/profiles_{year}.csv on first run.
    For the current season, the cache is automatically refreshed if it is
    older than 24 hours so new pitchers and updated metrics stay current.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"profiles_{year}.csv")

    is_current_year = (year == date.today().year)
    cache_exists = os.path.exists(path)

    if cache_exists and not force:
        if is_current_year and _cache_is_stale(path):
            print(f"  [{year}] Cache is >24 h old — refreshing current season...")
        else:
            df = pd.read_csv(path)
            print(f"  [{year}] Loaded {len(df)} profiles from cache.")
            return df

    print(f"  [{year}] Fetching Statcast data (this takes a few minutes)...")
    chunks = []
    for start, end in _months_for_year(year):
        try:
            df = statcast(start, end)
            if df is not None and not df.empty:
                chunks.append(df)
                print(f"    {start} to {end}: {len(df):,} pitches")
            else:
                print(f"    {start} to {end}: no data")
        except Exception as exc:
            print(f"    {start} to {end}: WARNING — {exc}")

    if not chunks:
        print(f"  [{year}] No data retrieved.")
        return pd.DataFrame()

    raw      = pd.concat(chunks, ignore_index=True)
    profiles = _aggregate_raw(raw, year)

    if profiles.empty:
        print(f"  [{year}] Aggregation returned no profiles.")
        return pd.DataFrame()

    profiles.to_csv(path, index=False)
    print(f"  [{year}] Saved {len(profiles)} profiles -> {path}")
    return profiles


def fetch_era_data(year: int, force: bool = False) -> pd.DataFrame:
    """
    Load or fetch pitcher ERA data for a season from Baseball Reference.
    Caches to data/era_{year}.csv.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"era_{year}.csv")

    is_current_year = (year == date.today().year)
    cache_exists = os.path.exists(path)

    if cache_exists and not force:
        if is_current_year and _cache_is_stale(path):
            print(f"  [{year}] ERA cache is >24 h old — refreshing...")
        else:
            return pd.read_csv(path)

    print(f"  [{year}] Fetching ERA data from Baseball Reference...")
    try:
        bref = pitching_stats_bref(year)
        keep = [c for c in ["Name", "ERA", "IP", "G", "GS"] if c in bref.columns]
        era_df = bref[keep].copy()
        era_df = era_df.rename(columns={"Name": "pitcher_name"})
        # De-duplicate multi-team players — keep the row with most IP (season total)
        era_df["IP"] = pd.to_numeric(era_df["IP"], errors="coerce")
        era_df = era_df.sort_values("IP", ascending=False).drop_duplicates("pitcher_name")
        era_df["year"] = year
        era_df.to_csv(path, index=False)
        print(f"  [{year}] Saved {len(era_df)} ERA rows -> {path}")
        return era_df
    except Exception as exc:
        print(f"  [{year}] WARNING: Could not fetch ERA — {exc}")
        return pd.DataFrame()


# ── Multi-season loaders ──────────────────────────────────────

def load_all_profiles(seasons=None, force: bool = False, cached_only: bool = False) -> pd.DataFrame:
    """
    Load and concatenate profiles for all (or specified) seasons.

    Parameters
    ----------
    cached_only : if True, only load seasons that already have a cache file.
                  Never triggers a Statcast fetch. Use for interactive lookups.
    """
    if seasons is None:
        seasons = range(STATCAST_FIRST_YEAR, STATCAST_LAST_YEAR + 1)
    dfs = []
    for year in seasons:
        path = os.path.join(DATA_DIR, f"profiles_{year}.csv")
        if cached_only and not os.path.exists(path):
            continue
        df = fetch_season_profiles(year, force=force)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_all_era(seasons=None) -> pd.DataFrame:
    """Load and concatenate ERA data for all (or specified) seasons."""
    if seasons is None:
        seasons = range(STATCAST_FIRST_YEAR, STATCAST_LAST_YEAR + 1)
    dfs = []
    for year in seasons:
        df = fetch_era_data(year)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def merge_era(profiles: pd.DataFrame, era_all: pd.DataFrame) -> pd.DataFrame:
    """
    Join ERA onto profiles by matching pitcher_name + year.
    Returns profiles with added 'era', 'fip', 'ip' columns where available.
    """
    if era_all.empty:
        profiles["era"] = np.nan
        return profiles

    era_cols = {c.lower(): c for c in era_all.columns}
    era_norm = era_all.copy()
    era_norm.columns = [c.lower() for c in era_norm.columns]
    era_norm = era_norm.rename(columns={"name": "pitcher_name"}) if "name" in era_norm.columns else era_norm

    merged = profiles.merge(
        era_norm[["pitcher_name", "year"] + [c for c in ["era", "fip", "ip"] if c in era_norm.columns]],
        on=["pitcher_name", "year"],
        how="left",
    )
    return merged
