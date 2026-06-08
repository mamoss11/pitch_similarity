# ─────────────────────────────────────────────────────────────
#  Pitch Similarity Model — CLI Entry Point
#
#  Usage:
#    python main.py --fetch 2025          # pull + cache one season
#    python main.py --fetch-all           # pull all seasons (slow first run)
#    python main.py --pitcher "Gerrit Cole" --year 2025
#    python main.py --pitcher "Gerrit Cole" --year 2025 --hand
# ─────────────────────────────────────────────────────────────
import argparse
import sys
import io

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from config import STATCAST_FIRST_YEAR, STATCAST_LAST_YEAR, PITCH_TYPE_NAMES, TOP_N_COMPS
from data_fetcher import load_all_profiles, load_all_era, fetch_season_profiles, merge_era
from similarity import normalize_features, get_arsenal_comps
from era_model import predict_era


def _load(args) -> pd.DataFrame:
    """
    Load cached profiles + ERA, merge, normalize.
    Only reads already-cached season files — never triggers a fetch.
    """
    seasons = list(range(STATCAST_FIRST_YEAR, STATCAST_LAST_YEAR + 1))
    print(f"\nLoading cached profiles ({seasons[0]}–{seasons[-1]})...")
    profiles = load_all_profiles(seasons, cached_only=True)

    if profiles.empty:
        print("No cached profiles found. Run --fetch-all (or --fetch YEAR) first.")
        sys.exit(1)

    cached_years = sorted(profiles["year"].unique())
    print(f"  Loaded seasons: {cached_years}")

    print(f"Loading ERA data...")
    era_all  = load_all_era(cached_years)
    profiles = merge_era(profiles, era_all)

    print(f"Normalizing features...")
    profiles_norm = normalize_features(profiles)
    return profiles_norm


def cmd_fetch(year: int, force: bool = False):
    print(f"\nFetching season {year}...")
    df = fetch_season_profiles(year, force=force)
    print(f"Done. {len(df)} pitch profiles cached.")


def cmd_show_pitcher(args, profiles_norm: pd.DataFrame):
    name = args.pitcher.strip()
    year = args.year

    # Find pitcher
    matches = profiles_norm[
        profiles_norm["pitcher_name"].str.contains(name, case=False, na=False) &
        (profiles_norm["year"] == year)
    ]
    if matches.empty:
        print(f"\nNo pitcher matching '{name}' found in {year}.")
        # Show close matches across all years
        all_matches = profiles_norm[
            profiles_norm["pitcher_name"].str.contains(name, case=False, na=False)
        ][["pitcher_name", "year"]].drop_duplicates().sort_values("year")
        if not all_matches.empty:
            print("Did you mean one of these?")
            print(all_matches.to_string(index=False))
        return

    pitcher_id   = matches["pitcher_id"].iloc[0]
    pitcher_name = matches["pitcher_name"].iloc[0]

    print(f"\n{'═'*65}")
    print(f"  {pitcher_name}  ({year})")
    print(f"{'═'*65}")

    # ── Arsenal ────────────────────────────────────────────────
    arsenal = matches[[
        "pitch_type", "n_pitches", "usage_pct",
        "velo", "ivb", "hb", "extension",
        "release_height", "release_side", "spin_rate",
    ]].copy()
    arsenal["pitch_name"] = arsenal["pitch_type"].map(
        lambda x: PITCH_TYPE_NAMES.get(x, x)
    )
    arsenal = arsenal.sort_values("usage_pct", ascending=False)

    print("\n  ARSENAL\n  " + "-"*62)
    print(f"  {'Pitch':<22} {'Usage':>6} {'Velo':>6} {'IVB':>6} {'HB':>6} "
          f"{'Ext':>5} {'Ht':>5} {'Spin':>6}")
    print("  " + "-"*62)
    for _, r in arsenal.iterrows():
        print(f"  {r['pitch_name']:<22} {r['usage_pct']:>5.1%} "
              f"{r['velo']:>6.1f} {r['ivb']:>6.1f} {r['hb']:>6.1f} "
              f"{r['extension']:>5.1f} {r['release_height']:>5.1f} "
              f"{r['spin_rate']:>6.0f}")

    # ── Predicted ERA ─────────────────────────────────────────
    era_result = predict_era(
        pitcher_id, year, profiles_norm,
        top_n=args.top_n, same_hand_only=args.hand,
    )
    if not np.isnan(era_result["predicted_era"]):
        print(f"\n  PREDICTED ERA:  {era_result['predicted_era']:.2f}  "
              f"(confidence: {era_result['confidence']:.0%}, "
              f"{era_result['n_comps']} total comps)")
        if "era" in matches.columns and matches["era"].notna().any():
            actual = matches["era"].iloc[0]
            if not np.isnan(actual):
                print(f"  ACTUAL ERA:     {actual:.2f}")

    # ── Comps per pitch ───────────────────────────────────────
    comps_all = get_arsenal_comps(
        pitcher_id, year, profiles_norm,
        top_n=args.top_n, same_hand_only=args.hand,
    )

    for pt, comp_df in sorted(comps_all.items(), key=lambda x: -matches[matches["pitch_type"]==x[0]]["usage_pct"].values[0]):
        pt_name = PITCH_TYPE_NAMES.get(pt, pt)
        usage   = matches[matches["pitch_type"] == pt]["usage_pct"].values[0]
        print(f"\n  {pt_name} ({pt}) — {usage:.1%} usage — Top {len(comp_df)} comps")
        print("  " + "-"*65)

        if comp_df.empty:
            print("  (no comps found)")
            continue

        era_col = "era" in comp_df.columns
        hdr = f"  {'Pitcher':<24} {'Yr':>4} {'Sim':>7} {'Velo':>6} {'IVB':>6} {'HB':>6}"
        if era_col:
            hdr += f"  {'ERA':>5}"
        print(hdr)
        print("  " + "-"*65)

        for _, r in comp_df.iterrows():
            line = (f"  {r['pitcher_name']:<24} {int(r['year']):>4} "
                    f"{r['similarity']:>7.1%} {r['velo']:>6.1f} "
                    f"{r['ivb']:>6.1f} {r['hb']:>6.1f}")
            if era_col and not np.isnan(r.get("era", np.nan)):
                line += f"  {r['era']:>5.2f}"
            print(line)

    print()


def main():
    parser = argparse.ArgumentParser(description="Pitch Similarity Model")
    parser.add_argument("--fetch",     type=int,  metavar="YEAR",
                        help="Fetch and cache one season")
    parser.add_argument("--fetch-all", action="store_true",
                        help=f"Fetch all seasons {STATCAST_FIRST_YEAR}–{STATCAST_LAST_YEAR}")
    parser.add_argument("--force",     action="store_true",
                        help="Re-fetch even if cache exists")
    parser.add_argument("--pitcher",   type=str,  help="Pitcher name to look up")
    parser.add_argument("--year",      type=int,  default=2025,
                        help="Season year (default: 2025)")
    parser.add_argument("--top-n",     type=int,  default=TOP_N_COMPS,
                        help=f"Number of comps to show (default: {TOP_N_COMPS})")
    parser.add_argument("--hand",      action="store_true",
                        help="Only return same-handedness comps")
    args = parser.parse_args()

    if args.fetch:
        cmd_fetch(args.fetch, force=args.force)
        return

    if args.fetch_all:
        for year in range(STATCAST_FIRST_YEAR, STATCAST_LAST_YEAR + 1):
            cmd_fetch(year, force=args.force)
        return

    if args.pitcher:
        profiles_norm = _load(args)
        cmd_show_pitcher(args, profiles_norm)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
