"""
phase5_championship.py
======================
Phase 5 features: dynamic team & driver championship strength.

All features use CUMULATIVE points from PRIOR rounds only (.shift(1) / before this race).
The current race's points are never included — zero leakage risk.

Features added:
  - driver_champ_points_before    (driver championship points entering this race)
  - constructor_champ_points_before (constructor points entering this race)
  - driver_rolling_form_5         (driver avg points per race, last 5 rounds)
  - constructor_rolling_form_5    (constructor avg points per race, last 5 rounds)
  - driver_champ_position_before  (driver championship standing entering this race)
  - constructor_champ_position_before
"""
import pandas as pd
import numpy as np

PHASE5_FEATURES = [
    "driver_champ_points_before",
    "constructor_champ_points_before",
    "driver_rolling_form_5",
    "constructor_rolling_form_5",
    "driver_champ_position_before",
    "constructor_champ_position_before",
]


def add_phase5_features(
    df: pd.DataFrame,
    driver_standings_df: pd.DataFrame,
    constructor_standings_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes and merges Phase 5 championship/form features.

    Parameters
    ----------
    df : pd.DataFrame
        Base dataframe with driverRef, constructor_name, year, round columns.
    driver_standings_df : pd.DataFrame
        Ergast driver_standings.csv (cumulative standings per round).
    constructor_standings_df : pd.DataFrame
        Ergast constructor_standings.csv.
    results_df : pd.DataFrame
        Ergast results.csv merged with races — needed for rolling form from actual pts.

    Returns
    -------
    df with Phase 5 columns appended.
    """
    for col in PHASE5_FEATURES:
        if col in df.columns:
            df = df.drop(columns=[col])

    # --- Driver championship points/position before this race ---
    # Ergast driver_standings gives cumulative points AFTER each round.
    # To get "before this race", we use the standings from round-1.
    ds = driver_standings_df.copy()
    ds["points"]   = pd.to_numeric(ds["points"], errors="coerce").fillna(0)
    ds["position"] = pd.to_numeric(ds["position"], errors="coerce")

    # Merge drivers table to get driverRef
    # The driver_standings table has driverId; we need driverRef from df
    # Strategy: merge standings onto results to get driverRef, then aggregate
    results_with_ref = results_df[["raceId", "driverId", "driverRef"]].drop_duplicates() \
        if "driverRef" in results_df.columns else pd.DataFrame()

    if not results_with_ref.empty and "driverId" in ds.columns:
        ds = ds.merge(results_with_ref, on=["raceId", "driverId"], how="left")

    # Also need year and round from races
    if "year" not in ds.columns and "raceId" in ds.columns:
        races_mini = results_df[["raceId", "year", "round"]].drop_duplicates() \
            if "year" in results_df.columns else pd.DataFrame()
        if not races_mini.empty:
            ds = ds.merge(races_mini, on="raceId", how="left")

    if "driverRef" in ds.columns and "year" in ds.columns and "round" in ds.columns:
        ds = ds.sort_values(["year", "round"])

        # Shift by 1 round to get "before this race" standings
        ds_shifted = ds.copy()
        ds_shifted["driver_champ_points_before"] = ds.groupby(
            ["driverRef", "year"]
        )["points"].shift(1).fillna(0)
        ds_shifted["driver_champ_position_before"] = ds.groupby(
            ["driverRef", "year"]
        )["position"].shift(1)

        ds_shifted = ds_shifted[["driverRef", "year", "round",
                                  "driver_champ_points_before",
                                  "driver_champ_position_before"]].drop_duplicates()
        df = df.merge(ds_shifted, on=["driverRef", "year", "round"], how="left")
        df["driver_champ_points_before"]   = df["driver_champ_points_before"].fillna(0)
        df["driver_champ_position_before"] = df["driver_champ_position_before"].fillna(20)
    else:
        df["driver_champ_points_before"]   = 0.0
        df["driver_champ_position_before"] = 20.0
        print("[Phase5] Warning: Could not compute driver championship features — check standings data.")

    # --- Constructor championship points/position before this race ---
    cs = constructor_standings_df.copy()
    cs["points"]   = pd.to_numeric(cs["points"], errors="coerce").fillna(0)
    cs["position"] = pd.to_numeric(cs["position"], errors="coerce")

    # Get constructor_name via constructorId
    results_constr = results_df[["raceId", "constructorId", "constructor_name"]].drop_duplicates() \
        if "constructor_name" in results_df.columns else pd.DataFrame()

    if not results_constr.empty and "constructorId" in cs.columns:
        cs = cs.merge(results_constr, on=["raceId", "constructorId"], how="left")

    if "year" not in cs.columns and "raceId" in cs.columns:
        races_mini = results_df[["raceId", "year", "round"]].drop_duplicates() \
            if "year" in results_df.columns else pd.DataFrame()
        if not races_mini.empty:
            cs = cs.merge(races_mini, on="raceId", how="left")

    if "constructor_name" in cs.columns and "year" in cs.columns and "round" in cs.columns:
        cs = cs.sort_values(["year", "round"])
        cs_shifted = cs.copy()
        cs_shifted["constructor_champ_points_before"] = cs.groupby(
            ["constructor_name", "year"]
        )["points"].shift(1).fillna(0)
        cs_shifted["constructor_champ_position_before"] = cs.groupby(
            ["constructor_name", "year"]
        )["position"].shift(1)

        cs_shifted = cs_shifted[["constructor_name", "year", "round",
                                  "constructor_champ_points_before",
                                  "constructor_champ_position_before"]].drop_duplicates()
        df = df.merge(cs_shifted, on=["constructor_name", "year", "round"], how="left")
        df["constructor_champ_points_before"]   = df["constructor_champ_points_before"].fillna(0)
        df["constructor_champ_position_before"] = df["constructor_champ_position_before"].fillna(10)
    else:
        df["constructor_champ_points_before"]   = 0.0
        df["constructor_champ_position_before"] = 10.0
        print("[Phase5] Warning: Could not compute constructor championship features.")

    # --- Rolling form (last 5 races avg points) ---
    # Compute from results_df actual points per driver/constructor per round
    if "points" in results_df.columns and "driverRef" in results_df.columns:
        rp = results_df[["driverRef", "year", "round", "points"]].copy()
        rp["points"] = pd.to_numeric(rp["points"], errors="coerce").fillna(0)
        rp = rp.sort_values(["year", "round"])

        rp["driver_rolling_form_5"] = rp.groupby("driverRef")["points"].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        ).fillna(0)

        df = df.merge(
            rp[["driverRef", "year", "round", "driver_rolling_form_5"]].drop_duplicates(),
            on=["driverRef", "year", "round"],
            how="left"
        )
        df["driver_rolling_form_5"] = df["driver_rolling_form_5"].fillna(0)
    else:
        df["driver_rolling_form_5"] = 0.0

    if "points" in results_df.columns and "constructor_name" in results_df.columns:
        cp = results_df[["constructor_name", "year", "round", "points"]].copy()
        cp["points"] = pd.to_numeric(cp["points"], errors="coerce").fillna(0)
        cp = cp.sort_values(["year", "round"])

        # Sum constructor points (both drivers) per round first
        cp_sum = cp.groupby(["constructor_name", "year", "round"])["points"].sum().reset_index()
        cp_sum["constructor_rolling_form_5"] = cp_sum.groupby("constructor_name")["points"].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        ).fillna(0)

        df = df.merge(
            cp_sum[["constructor_name", "year", "round", "constructor_rolling_form_5"]].drop_duplicates(),
            on=["constructor_name", "year", "round"],
            how="left"
        )
        df["constructor_rolling_form_5"] = df["constructor_rolling_form_5"].fillna(0)
    else:
        df["constructor_rolling_form_5"] = 0.0

    return df
