"""
phase1_practice_pace.py
=======================
Phase 1 features: practice session lap times and qualifying sector times.

All data is from pre-race sessions (FP1, FP2, FP3, Qualifying).
No race-time information is used — zero leakage risk.

Features added:
  - fp1_best_laptime_s
  - fp2_best_laptime_s
  - fp3_best_laptime_s
  - fp2_longrun_avg_s        (race-pace proxy)
  - fp2_longrun_laps
  - quali_sector1_s
  - quali_sector2_s
  - quali_sector3_s
  - quali_pole_delta_s       (gap to pole in seconds)
"""
import os
import sys
import pandas as pd
import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.fastf1_loader import (
    fetch_session_best_laps,
    fetch_fp2_longrun_pace,
    fetch_qualifying_sectors,
)

PHASE1_FEATURES = [
    "fp1_best_laptime_s",
    "fp2_best_laptime_s",
    "fp3_best_laptime_s",
    "fp2_longrun_avg_s",
    "fp2_longrun_laps",
    "quali_sector1_s",
    "quali_sector2_s",
    "quali_sector3_s",
    "quali_pole_delta_s",
]

# Driver abbreviation → Ergast driverRef mapping
# FastF1 uses 3-letter codes; Ergast uses full references.
# This mapping covers the modern era (2018+). Add entries as needed.
DRIVER_CODE_TO_REF = {
    "VER": "verstappen",
    "HAM": "hamilton",
    "NOR": "norris",
    "LEC": "leclerc",
    "SAI": "sainz",
    "RUS": "russell",
    "PER": "perez",
    "ALO": "alonso",
    "STR": "stroll",
    "OCO": "ocon",
    "GAS": "gasly",
    "TSU": "tsunoda",
    "BOT": "bottas",
    "ZHO": "zhou",
    "MAG": "magnussen",
    "HUL": "hulkenberg",
    "ALB": "albon",
    "SAR": "sargeant",
    "RIC": "ricciardo",
    "LAW": "lawson",
    "BEA": "bearman",
    "DOO": "doohan",
    "ANT": "antonelli",
    "HAD": "hadjar",
    "BOR": "bortoleto",
    "OCO": "ocon",
    "GIO": "giovinazzi",
    "RAI": "raikkonen",
    "VET": "vettel",
    "MSC": "mick_schumacher",
    "MAZ": "mazepin",
    "GRO": "grosjean",
    "KVY": "kvyat",
    "GIO": "giovinazzi",
    "LAT": "latifi",
    "FIT": "fittipaldi",
    "AIT": "aitken",
    "DEV": "de_vries",
    "PIA": "piastri",
    "COL": "colapinto",
}


def build_phase1_features(years: list, gp_map: dict) -> pd.DataFrame:
    """
    Fetches and assembles Phase 1 features for all (year, gp) combinations.

    Parameters
    ----------
    years : list of int
    gp_map : dict
        Maps (year, round_number) → gp_name_for_fastf1.
        Example: {(2023, 1): "Bahrain", (2023, 2): "Saudi Arabia", ...}

    Returns
    -------
    DataFrame with driverRef, year, round, and all PHASE1_FEATURES.
    """
    all_rows = []

    for (year, rnd), gp_name in gp_map.items():
        print(f"[Phase1] Fetching {year} Round {rnd} — {gp_name}")

        fp1 = fetch_session_best_laps(year, gp_name, "FP1")
        fp2 = fetch_session_best_laps(year, gp_name, "FP2")
        fp3 = fetch_session_best_laps(year, gp_name, "FP3")
        fp2_lr = fetch_fp2_longrun_pace(year, gp_name)
        q_sec = fetch_qualifying_sectors(year, gp_name)

        # Merge on driver_code
        merged = pd.DataFrame({"driver_code": list(DRIVER_CODE_TO_REF.keys())})

        def _join(merged, df, suffix):
            if df.empty:
                return merged
            df = df.rename(columns={"session_best_laptime_s": suffix})
            return merged.merge(df[["driver_code", suffix]], on="driver_code", how="left")

        merged = _join(merged, fp1, "fp1_best_laptime_s")
        merged = _join(merged, fp2, "fp2_best_laptime_s")
        merged = _join(merged, fp3, "fp3_best_laptime_s")

        if not fp2_lr.empty:
            merged = merged.merge(
                fp2_lr[["driver_code", "fp2_longrun_avg_s", "fp2_longrun_laps"]],
                on="driver_code", how="left"
            )
        else:
            merged["fp2_longrun_avg_s"] = np.nan
            merged["fp2_longrun_laps"] = np.nan

        if not q_sec.empty:
            merged = merged.merge(
                q_sec[["driver_code", "quali_sector1_s", "quali_sector2_s",
                        "quali_sector3_s", "quali_pole_delta_s"]],
                on="driver_code", how="left"
            )
        else:
            for col in ["quali_sector1_s", "quali_sector2_s", "quali_sector3_s", "quali_pole_delta_s"]:
                merged[col] = np.nan

        merged["driverRef"] = merged["driver_code"].map(DRIVER_CODE_TO_REF)
        merged["year"] = year
        merged["round"] = rnd

        # Drop rows where driverRef is unknown
        merged = merged.dropna(subset=["driverRef"])
        all_rows.append(merged)

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    result = result.drop(columns=["driver_code"], errors="ignore")
    return result


def add_phase1_to_df(base_df: pd.DataFrame, phase1_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merges Phase 1 features onto the base training dataframe.
    Rows without FastF1 data (pre-2018) get NaN — imputed later.
    """
    if phase1_df.empty:
        for col in PHASE1_FEATURES:
            base_df[col] = np.nan
        return base_df

    merged = base_df.merge(
        phase1_df[["driverRef", "year", "round"] + PHASE1_FEATURES],
        on=["driverRef", "year", "round"],
        how="left"
    )
    return merged
