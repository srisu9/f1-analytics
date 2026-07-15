"""
phase4_telemetry.py
===================
Phase 4 features: telemetry-derived aggregates from practice sessions.

RAW TELEMETRY IS NOT FED TO THE MODEL.
Each telemetry signal is reduced to a single scalar per driver per race weekend.

All data is from FP2 (race-pace simulation session) or Qualifying.
Strictly pre-race — zero leakage risk.

Features added:
  - avg_speed_kph       (mean speed over best practice laps)
  - top_speed_kph       (max speed recorded in FP2/FP3)
  - throttle_pct        (% of telemetry samples at full throttle ≥98%)
  - brake_pct           (% of samples with brake applied)
  - sector1_std_s       (lap-to-lap sector 1 consistency in qualifying)
  - sector2_std_s
  - sector3_std_s
  - tire_deg_estimate   (lap-time degradation rate in FP2 long run, s/lap)
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.fastf1_loader import (
    fetch_telemetry_aggregates,
    fetch_sector_consistency,
    fetch_fp2_longrun_pace,
    fetch_session_best_laps,
)

PHASE4_FEATURES = [
    "avg_speed_kph",
    "top_speed_kph",
    "throttle_pct",
    "brake_pct",
    "sector1_std_s",
    "sector2_std_s",
    "sector3_std_s",
    "tire_deg_estimate",
]

# Reuse driver code map from phase1
from src.features.phase1_practice_pace import DRIVER_CODE_TO_REF


def _compute_tire_degradation(year: int, gp_name: str, min_laps: int = 6) -> pd.DataFrame:
    """
    Estimates tire degradation rate from FP2 long runs.
    Computes the slope of lap time vs lap number within the longest stint.

    Returns
    -------
    DataFrame: driver_code, tire_deg_estimate (seconds lost per lap due to deg)
    """
    if not _ff1_available():
        return pd.DataFrame()
    try:
        import fastf1
        session = fastf1.get_session(year, gp_name, "FP2")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps.pick_quicklaps().copy()
        laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

        results = []
        for driver, grp in laps.groupby("Driver"):
            grp = grp.sort_values("LapNumber").reset_index(drop=True)
            grp["stint"] = (grp["Compound"] != grp["Compound"].shift()).cumsum()
            for _, stint_df in grp.groupby("stint"):
                if len(stint_df) >= min_laps:
                    x = np.arange(len(stint_df))
                    y = stint_df["LapTime_s"].values
                    if np.std(y) > 0:
                        slope, _ = np.polyfit(x, y, 1)
                        results.append({
                            "driver_code": driver,
                            "tire_deg_estimate": max(slope, 0),  # negative slope = improvement, clip to 0
                            "stint_laps": len(stint_df)
                        })
            # Break after first qualifying long stint per driver
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results).sort_values("stint_laps", ascending=False)
        df = df.drop_duplicates(subset="driver_code", keep="first")
        return df[["driver_code", "tire_deg_estimate"]]

    except Exception as exc:
        print(f"[Phase4] Tire deg error {year} {gp_name}: {exc}")
        return pd.DataFrame()


def _ff1_available() -> bool:
    try:
        import fastf1
        return True
    except ImportError:
        return False


def build_phase4_features(year_gp_map: dict) -> pd.DataFrame:
    """
    Assembles Phase 4 telemetry features for all (year, round) → gp_name pairs.
    """
    all_rows = []

    for (year, rnd), gp_name in year_gp_map.items():
        print(f"[Phase4] Telemetry features for {year} R{rnd} — {gp_name}")

        # Telemetry aggregates from FP2
        tel_df = fetch_telemetry_aggregates(year, gp_name, session_type="FP2")

        # Sector consistency from Qualifying
        sec_df = fetch_sector_consistency(year, gp_name, session_type="Q")

        # Tire degradation from FP2
        deg_df = _compute_tire_degradation(year, gp_name)

        # Merge all on driver_code
        base = pd.DataFrame({"driver_code": list(DRIVER_CODE_TO_REF.keys())})

        if not tel_df.empty:
            base = base.merge(
                tel_df[["driver_code", "avg_speed_kph", "top_speed_kph", "throttle_pct", "brake_pct"]],
                on="driver_code", how="left"
            )
        else:
            for col in ["avg_speed_kph", "top_speed_kph", "throttle_pct", "brake_pct"]:
                base[col] = np.nan

        if not sec_df.empty:
            base = base.merge(
                sec_df[["driver_code", "sector1_std_s", "sector2_std_s", "sector3_std_s"]],
                on="driver_code", how="left"
            )
        else:
            for col in ["sector1_std_s", "sector2_std_s", "sector3_std_s"]:
                base[col] = np.nan

        if not deg_df.empty:
            base = base.merge(deg_df, on="driver_code", how="left")
        else:
            base["tire_deg_estimate"] = np.nan

        base["driverRef"] = base["driver_code"].map(DRIVER_CODE_TO_REF)
        base["year"] = year
        base["round"] = rnd
        base = base.dropna(subset=["driverRef"])
        all_rows.append(base)

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    result = result.drop(columns=["driver_code"], errors="ignore")
    return result


def add_phase4_to_df(base_df: pd.DataFrame, phase4_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merges Phase 4 features onto the base training dataframe.
    Pre-2018 rows get NaN (expected — imputed as circuit median in training).
    """
    if phase4_df.empty:
        for col in PHASE4_FEATURES:
            base_df[col] = np.nan
        return base_df

    for col in PHASE4_FEATURES:
        if col in base_df.columns:
            base_df = base_df.drop(columns=[col])

    merged = base_df.merge(
        phase4_df[["driverRef", "year", "round"] + PHASE4_FEATURES],
        on=["driverRef", "year", "round"],
        how="left"
    )
    return merged
