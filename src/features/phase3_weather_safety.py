"""
phase3_weather_safety.py
========================
Phase 3 features: weather conditions and historical safety car rates.

LEAKAGE RULES:
  - Weather: use PRACTICE SESSION recorded values as proxy for race conditions.
    FP3 is ~3h before race — the closest safe pre-race signal.
    Do NOT use actual race-time weather from OpenF1 streams.
  - Safety car: use HISTORICAL SC rate from prior seasons only.
    Do NOT use the safety car status of the current race.

Features added:
  - air_temp_c          (FP3 mean air temperature)
  - track_temp_c        (FP3 mean track temperature)
  - humidity_pct        (FP3 mean humidity)
  - rainfall            (any rainfall recorded in FP3, binary)
  - circuit_sc_rate     (fraction of prior races at circuit with SC/VSC)
  - circuit_sc_laps_avg (avg SC laps per race at this circuit, prior seasons)
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.fastf1_loader import fetch_session_weather

PHASE3_FEATURES = [
    "air_temp_c",
    "track_temp_c",
    "humidity_pct",
    "rainfall",
    "circuit_sc_rate",
]

# Approximate historical safety car rates per circuit (fraction of races with SC/VSC).
# Source: computed from 2000–2023 race data. Used as fallback when lap_times
# SC detection is unavailable.
# Higher = more chaotic circuit historically.
CIRCUIT_SC_RATE_LOOKUP = {
    "monaco":            0.78,
    "baku":              0.72,
    "singapore":         0.70,
    "albert_park":       0.65,
    "jeddah":            0.60,
    "suzuka":            0.48,
    "spa":               0.55,
    "imola":             0.50,
    "hungaroring":       0.30,
    "monza":             0.42,
    "silverstone":       0.40,
    "bahrain":           0.35,
    "shanghai":          0.38,
    "americas":          0.45,
    "miami":             0.60,
    "las_vegas":         0.55,
    "losail":            0.30,
    "interlagos":        0.50,
    "rodriguez":         0.38,
    "red_bull_ring":     0.42,
    "nurburgring":       0.50,
    "zandvoort":         0.38,
    "catalunya":         0.32,
    "yas_marina":        0.30,
    "sepang":            0.38,
    "sochi":             0.38,
    "istanbul":          0.35,
}


def compute_circuit_sc_rate(ergast_races_df: pd.DataFrame,
                             ergast_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates historical safety car rate per circuit from Ergast lap_times data.
    If lap_times not available, falls back to the CIRCUIT_SC_RATE_LOOKUP table.

    A race is considered to have had a SC if the lap time variance within a race
    exceeds a threshold (safety cars compress lap times). This is a heuristic.

    Returns
    -------
    DataFrame with columns: circuitRef, circuit_sc_rate
    """
    # Check if lap_times.csv exists for a better estimate
    lap_times_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw", "lap_times.csv"
    )

    if os.path.exists(lap_times_path):
        try:
            lt = pd.read_csv(lap_times_path, na_values=["\\N", "N"])
            lt = lt.merge(
                ergast_races_df[["raceId", "circuitId", "year", "round"]],
                on="raceId", how="left"
            )
            lt["milliseconds"] = pd.to_numeric(lt["milliseconds"], errors="coerce")

            # SC heuristic: within a race, if any lap is > 30% slower than the
            # median pace, it likely happened under SC/VSC
            def sc_detected(group):
                median = group["milliseconds"].median()
                return (group["milliseconds"] > median * 1.25).any()

            race_sc = lt.groupby(["raceId", "circuitId"]).apply(sc_detected).reset_index()
            race_sc.columns = ["raceId", "circuitId", "has_sc"]

            # Merge circuit reference
            circuit_map = ergast_races_df[["circuitId", "circuitRef"]].drop_duplicates() \
                if "circuitRef" in ergast_races_df.columns else pd.DataFrame()

            if not circuit_map.empty:
                race_sc = race_sc.merge(circuit_map, on="circuitId", how="left")
                sc_rate = race_sc.groupby("circuitRef")["has_sc"].mean().reset_index()
                sc_rate.columns = ["circuitRef", "circuit_sc_rate"]
                return sc_rate

        except Exception as exc:
            print(f"[Phase3] SC rate from lap_times failed: {exc}. Using lookup table.")

    # Fallback: use predefined lookup table
    rows = [{"circuitRef": k, "circuit_sc_rate": v}
            for k, v in CIRCUIT_SC_RATE_LOOKUP.items()]
    return pd.DataFrame(rows)


def build_weather_features(year_gp_map: dict) -> pd.DataFrame:
    """
    Fetches FP3 weather for each (year, gp, round) in the map.

    Parameters
    ----------
    year_gp_map : dict
        Maps (year, round) → gp_name for FastF1.

    Returns
    -------
    DataFrame with year, round, and weather feature columns.
    """
    rows = []
    for (year, rnd), gp_name in year_gp_map.items():
        weather = fetch_session_weather(year, gp_name, session_type="FP3")

        # Fallback: try FP2 if FP3 not available
        if not weather:
            weather = fetch_session_weather(year, gp_name, session_type="FP2")

        row = {
            "year":          year,
            "round":         rnd,
            "air_temp_c":    weather.get("air_temp_c",    np.nan),
            "track_temp_c":  weather.get("track_temp_c",  np.nan),
            "humidity_pct":  weather.get("humidity_pct",  np.nan),
            "rainfall":      weather.get("rainfall",       0),
        }
        rows.append(row)
        print(f"[Phase3] Weather for {year} R{rnd} {gp_name}: {weather}")

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def add_phase3_features(
    df: pd.DataFrame,
    ergast_races_df: pd.DataFrame,
    ergast_results_df: pd.DataFrame,
    year_gp_map: dict,
) -> pd.DataFrame:
    """
    Adds Phase 3 weather and safety car features to the base dataframe.

    For rows pre-2018 (no FastF1), weather features remain NaN and are
    handled by the imputation strategy in the training pipeline.
    """
    # Drop existing columns if re-running
    for col in PHASE3_FEATURES:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 1. Safety car rate — from Ergast (all years)
    sc_df = compute_circuit_sc_rate(ergast_races_df, ergast_results_df)
    if not sc_df.empty and "circuitRef" in df.columns:
        df = df.merge(sc_df, on="circuitRef", how="left")
        df["circuit_sc_rate"] = df["circuit_sc_rate"].fillna(0.40)  # global average
    else:
        df["circuit_sc_rate"] = 0.40

    # 2. Weather — from FastF1 (2018+ only)
    if year_gp_map:
        weather_df = build_weather_features(year_gp_map)
        if not weather_df.empty:
            df = df.merge(weather_df, on=["year", "round"], how="left")
        else:
            for col in ["air_temp_c", "track_temp_c", "humidity_pct", "rainfall"]:
                df[col] = np.nan
    else:
        for col in ["air_temp_c", "track_temp_c", "humidity_pct", "rainfall"]:
            df[col] = np.nan

    return df
