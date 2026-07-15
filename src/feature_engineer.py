import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import GLOBAL_TOP10_PRIOR, ROLLING_AVG_FILL


def engineer_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all pre-race features from the raw merged Ergast dataframe.

    IMPORTANT: Must be called on the RAW merged dataframe (before clean_data),
    because it needs ``positionOrder`` and ``points`` to compute rolling stats
    and cumulative season points, then drops those columns itself.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Output of ``merge_datasets()`` — contains all raw Ergast columns.

    Returns
    -------
    pd.DataFrame
        Engineered feature dataframe with leakage columns removed.

    Raises
    ------
    ValueError
        If ``positionOrder`` or ``points`` columns are missing (indicates
        the function was called after ``clean_data``, which drops them).
    """
    df = merged_df.copy()

    # Guard: ensure raw race-result columns are present for feature computation
    if "positionOrder" not in df.columns:
        raise ValueError(
            "engineer_features() requires 'positionOrder' to be present. "
            "Call this function on the raw merged dataframe BEFORE clean_data()."
        )
    if "points" not in df.columns:
        raise ValueError(
            "engineer_features() requires 'points' to compute "
            "constructor_season_points. Call BEFORE clean_data()."
        )

    # ── Sort by date ──────────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(by=["date", "round"]).reset_index(drop=True)

    # ── Target variable ───────────────────────────────────────────────────────
    df["Top10"] = (df["positionOrder"] <= 10).astype(int)

    # ── Driver age ───────────────────────────────────────────────────────────
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["driver_age"] = (df["date"] - df["dob"]).dt.days / 365.25

    # ── Driver experience (career starts before this race) ───────────────────
    # cumcount() gives the number of previous rows for this driver (0-indexed),
    # which equals the number of races started before the current one.
    df["driver_experience"] = df.groupby("driverRef").cumcount()

    # ── Driver Top-10 rate (previously called driver_win_rate) ───────────────
    # Computed using only past races (.shift(1) prevents leakage).
    # Fill at 0: the cumulative count is 0 for the first race — avoid /0.
    # Default = GLOBAL_TOP10_PRIOR (0.50) for first-race drivers.
    df["driver_prev_top10_sum"]   = df.groupby("driverRef")["Top10"].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    df["driver_prev_races_count"] = df.groupby("driverRef").cumcount()
    df["driver_top10_rate"] = (
        df["driver_prev_top10_sum"]
        / df["driver_prev_races_count"].replace(0, np.nan)
    )
    df["driver_top10_rate"] = df["driver_top10_rate"].fillna(GLOBAL_TOP10_PRIOR)

    # ── Constructor Top-10 rate (previously called constructor_win_rate) ──────
    df["constructor_prev_top10_sum"]   = df.groupby("constructor_name")["Top10"].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    df["constructor_prev_races_count"] = df.groupby("constructor_name").cumcount()
    df["constructor_top10_rate"] = (
        df["constructor_prev_top10_sum"]
        / df["constructor_prev_races_count"].replace(0, np.nan)
    )
    df["constructor_top10_rate"] = df["constructor_top10_rate"].fillna(GLOBAL_TOP10_PRIOR)

    # ── Rolling average finish (3-race and 5-race windows) ───────────────────
    # Uses .shift(1) to exclude the current race.  min_periods=1 ensures we
    # get a value after the first race.  NaN (first race) filled with mid-field.
    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["rolling_avg_finish_3"] = df.groupby("driverRef")["positionOrder"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    ).fillna(ROLLING_AVG_FILL)
    df["rolling_avg_finish_5"] = df.groupby("driverRef")["positionOrder"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    ).fillna(ROLLING_AVG_FILL)

    # ── Previous race finish ──────────────────────────────────────────────────
    df["prev_race_finish"] = df.groupby("driverRef")["positionOrder"].shift(1).fillna(ROLLING_AVG_FILL)

    # ── Home race flag ────────────────────────────────────────────────────────
    nationality_to_country = {
        "British":        "UK",
        "German":         "Germany",
        "French":         "France",
        "Italian":        "Italy",
        "Spanish":        "Spain",
        "Australian":     "Australia",
        "Austrian":       "Austria",
        "Japanese":       "Japan",
        "Brazilian":      "Brazil",
        "Canadian":       "Canada",
        "American":       "USA",
        "Belgian":        "Belgium",
        "Dutch":          "Netherlands",
        "Monaco":         "Monaco",
        "Monegasque":     "Monaco",   # added: Leclerc
        "Mexican":        "Mexico",
        "Finnish":        "Finland",
        "Swiss":          "Switzerland",
        "Russian":        "Russia",
        "Swedish":        "Sweden",
        "New Zealander":  "New Zealand",
        "South African":  "South Africa",
        "Danish":         "Denmark",    # added: Magnussen
        "Thai":           "Thailand",   # added: Albon
        "Colombian":      "Colombia",   # added: Gutierrez era
        "Chinese":        "China",      # added: Zhou
        "Polish":         "Poland",
        "Argentine":      "Argentina",
        "Hungarian":      "Hungary",
        "Venezuelan":     "Venezuela",
        "Indonesian":     "Indonesia",
    }
    driver_country = df["driver_nationality"].map(nationality_to_country)
    df["home_race"] = (driver_country == df["country"]).fillna(False).astype(int)

    # ── Grid vs qualifying position difference ────────────────────────────────
    # grid_qualifying_diff > 0  → driver starts further back than qualified
    # grid_qualifying_diff < 0  → promoted (e.g., penalties for others)
    # grid_qualifying_diff = 0  → no change, or pit-lane start (NaN fallback)
    df["qualifying_position"] = pd.to_numeric(df["qualifying_position"], errors="coerce")
    df["grid"]                = pd.to_numeric(df["grid"],                errors="coerce")
    # For pit-lane starters: qualifying_position is NaN → impute with grid so diff=0.
    # NOTE: this means pit-lane starters are indistinguishable from "exactly qualified"
    # drivers — a known minor limitation documented in the engineering audit.
    df["qualifying_position_imputed"] = df["qualifying_position"].fillna(df["grid"])
    df["grid_qualifying_diff"]        = df["grid"] - df["qualifying_position_imputed"]

    # ── Constructor season points (cumulative, before current race) ───────────
    # Sum both drivers' points per team per round, then cumsum().shift(1).
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)
    round_pts = (
        df.groupby(["year", "round", "constructor_name"])["points"]
        .sum()
        .reset_index()
        .sort_values(["year", "round"])
    )
    round_pts["constructor_season_points"] = round_pts.groupby(
        ["year", "constructor_name"]
    )["points"].transform(lambda x: x.cumsum().shift(1).fillna(0.0))
    round_pts.drop(columns=["points"], inplace=True)
    df = pd.merge(df, round_pts, on=["year", "round", "constructor_name"], how="left")
    df["constructor_season_points"] = df["constructor_season_points"].fillna(0.0)

    # ── Drop helper / leakage / ID columns ───────────────────────────────────
    drop_helpers = [
        "driver_prev_top10_sum", "driver_prev_races_count",
        "constructor_prev_top10_sum", "constructor_prev_races_count",
        "qualifying_position_imputed",
    ]
    leakage_cols = [
        "position", "positionOrder", "positionText", "points",
        "time", "milliseconds", "fastestLap", "rank",
        "fastestLapTime", "fastestLapSpeed", "laps", "statusId",
    ]
    id_cols = ["resultId", "raceId", "driverId", "constructorId", "circuitId"]
    details_cols = [
        "fp1_date", "fp1_time", "fp2_date", "fp2_time",
        "fp3_date", "fp3_time", "sprint_date", "sprint_time",
        "quali_date", "quali_time", "race_time",
        "driver_url", "constructor_url", "race_url", "circuit_url",
        "constructorRef", "quali_car_number", "driver_number",
        "q1", "q2", "q3", "code", "forename", "surname", "dob",
    ]
    df.drop(
        columns=drop_helpers + leakage_cols + id_cols + details_cols,
        inplace=True, errors="ignore"
    )

    return df
