"""
phase2_circuit_history.py
=========================
Phase 2 features — Bayesian-smoothed circuit history and overtaking index.

Features (all strictly pre-race, zero leakage):
  1. smoothed_circuit_avg_finish  — Bayesian-smoothed driver avg finish at this circuit
  2. circuit_grid_finish_corr     — Rolling 3-year grid-to-finish correlation per circuit

Bayesian Smoothing Formula:
    smoothed = (n_visits * circuit_avg + K * career_avg) / (n_visits + K)

where:
  n_visits   = number of prior visits to this circuit
  circuit_avg = raw avg finishing position at this circuit (prior visits only)
  K          = smoothing factor (default 4)
  career_avg = driver's career-wide average finishing position (prior races only)

The career_avg prior is computed from ALL prior races (not just this circuit),
ensuring the Bayesian prior is a genuine global career average.
"""

import pandas as pd
import numpy as np

PHASE2_FEATURES = [
    "smoothed_circuit_avg_finish",
    "circuit_grid_finish_corr",
]


def add_phase2_features(df: pd.DataFrame, results_raw: pd.DataFrame, K: int = 4) -> pd.DataFrame:
    """
    Computes Phase 2 circuit history features and merges them onto df.

    Parameters
    ----------
    df : pd.DataFrame
        Base engineered dataframe (output of engineer_features).
    results_raw : pd.DataFrame
        Raw merged Ergast dataframe with positionOrder, grid, driverRef,
        circuitRef, year, round, date columns.
    K : int
        Bayesian smoothing factor. Higher K pulls estimates more toward
        the career prior when circuit data is sparse.

    Returns
    -------
    pd.DataFrame
        df with smoothed_circuit_avg_finish and circuit_grid_finish_corr added.
    """
    r = results_raw.copy()
    r["date"]          = pd.to_datetime(r["date"],          errors="coerce")
    r["positionOrder"] = pd.to_numeric(r["positionOrder"],  errors="coerce")
    r["grid"]          = pd.to_numeric(r["grid"],           errors="coerce")
    r = r.sort_values(["date", "round"]).reset_index(drop=True)

    # ── Step 1: Compute global career average finish per driver (prior races only)
    # This is computed over ALL races in r (not filtered to a specific circuit),
    # giving a genuine career-wide prior for the Bayesian formula.
    # cumcount() gives "races completed before this row" = n prior races.
    r["career_races"]      = r.groupby("driverRef").cumcount()
    r["career_finish_sum"] = r.groupby("driverRef")["positionOrder"].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    # career_avg = sum of prior finishes / n prior races.
    # For first race (career_races=0), use default mid-field of 12.0.
    r["career_avg_finish"] = (
        r["career_finish_sum"] / r["career_races"].replace(0, np.nan)
    ).fillna(12.0)

    # ── Step 2: Compute Bayesian-smoothed circuit-specific average finish ──────
    # For each (driver, circuit) group, compute cumulative stats sorted by date,
    # then apply Bayesian smoothing using the global career_avg as prior.
    driver_circuit_rows = []

    for (driver, circuit), grp in r.groupby(["driverRef", "circuitRef"]):
        grp = grp.sort_values("date").reset_index(drop=True)

        # n_visits[i] = number of prior visits to this circuit before row i
        grp["n_visits"] = grp.index  # 0, 1, 2, … (0 = first ever visit)

        # circuit_finish_sum[i] = sum of finishing positions in PRIOR visits
        grp["circuit_finish_sum"] = grp["positionOrder"].cumsum().shift(1).fillna(0)

        # Bayesian prior: career_avg for row i from the GLOBAL computation above.
        # These values come from the full r dataframe (all circuits), so this is
        # a genuine global career prior — not accidentally circuit-specific.
        career_prior = grp["career_avg_finish"].values  # already shifted via cumsum above

        grp["smoothed_circuit_avg_finish"] = (
            (grp["circuit_finish_sum"] + K * career_prior)
            / (grp["n_visits"] + K)
        )

        driver_circuit_rows.append(
            grp[["driverRef", "circuitRef", "year", "round", "smoothed_circuit_avg_finish"]]
        )

    dc_df = (
        pd.concat(driver_circuit_rows, ignore_index=True)
        if driver_circuit_rows
        else pd.DataFrame()
    )

    # ── Step 3: Circuit-level Overtaking Index (grid-finish correlation) ──────
    # For each circuit × year, compute Pearson correlation between grid position
    # and finishing position.  Then rolling-mean over 3 prior years (.shift(1)
    # prevents the current year's data from being used).
    # Granularity is year-level (not round), so all rounds of a year share the
    # same overtaking index value computed from prior years' data.
    circuit_years = (
        r.groupby(["circuitRef", "year"])
        .apply(
            lambda x: x["grid"].corr(x["positionOrder"]) if len(x) > 5 else np.nan,
            include_groups=False
        )
        .reset_index()
        .rename(columns={0: "grid_finish_corr"})
    )
    circuit_years = circuit_years.sort_values("year")
    circuit_years["circuit_grid_finish_corr"] = circuit_years.groupby("circuitRef")[
        "grid_finish_corr"
    ].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    ).fillna(0.50)  # 0.50 = neutral correlation (no prior data)

    # ── Step 4: Drop existing columns then merge new ones ────────────────────
    for col in PHASE2_FEATURES:
        if col in df.columns:
            df = df.drop(columns=[col])

    if not dc_df.empty:
        df = df.merge(
            dc_df,
            on=["driverRef", "circuitRef", "year", "round"],
            how="left"
        )
        df["smoothed_circuit_avg_finish"] = df["smoothed_circuit_avg_finish"].fillna(12.0)
    else:
        df["smoothed_circuit_avg_finish"] = 12.0

    if not circuit_years.empty:
        df = df.merge(
            circuit_years[["circuitRef", "year", "circuit_grid_finish_corr"]].drop_duplicates(),
            on=["circuitRef", "year"],
            how="left"
        )
        df["circuit_grid_finish_corr"] = df["circuit_grid_finish_corr"].fillna(0.50)
    else:
        df["circuit_grid_finish_corr"] = 0.50

    return df
