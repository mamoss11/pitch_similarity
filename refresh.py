# ─────────────────────────────────────────────────────────────
#  Pitch Similarity Model — Season Refresh
#
#  Re-fetches the current season's Statcast profiles and ERA data,
#  overwriting the existing cache. Run this daily/weekly to pick
#  up new pitchers and updated pitch metrics.
#
#  Usage:
#    python refresh.py              # refresh current season
#    python refresh.py --year 2025  # force-refresh a specific season
# ─────────────────────────────────────────────────────────────
import sys
import os
import argparse
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher import fetch_season_profiles, fetch_era_data


def main():
    parser = argparse.ArgumentParser(description="Refresh pitch similarity data")
    parser.add_argument(
        "--year", type=int, default=None,
        help="Season to refresh (default: current year)",
    )
    args = parser.parse_args()

    year = args.year or date.today().year

    # No data before March; season is fully done after October
    today = date.today()
    if year == today.year and today.month < 3:
        print("Off-season — MLB hasn't started yet. Nothing to refresh.")
        return

    print(f"\nRefreshing {year} season data...")
    print("-" * 40)

    profiles = fetch_season_profiles(year, force=True)
    print(f"Profiles: {len(profiles)} pitcher x pitch-type rows cached.")

    era = fetch_era_data(year, force=True)
    print(f"ERA data: {len(era)} pitcher rows cached.")

    print(f"\nDone. Reload the Streamlit app to see updated data.")


if __name__ == "__main__":
    main()
