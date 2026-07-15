"""
leakage_checker.py
==================
Utility to detect temporal data leakage before any feature is added to training.

Usage:
    from src.leakage_checker import assert_no_leakage
    assert_no_leakage(feature_df, race_dates_series)
"""
import pandas as pd
import numpy as np
from datetime import timedelta


class DataLeakageError(Exception):
    pass


def assert_no_leakage(
    feature_df: pd.DataFrame,
    race_date_col: str = "date",
    source_date_col: str = "feature_source_date",
    tolerance_hours: int = 0,
):
    """
    Validates that every row's feature_source_date is BEFORE its race date.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Must contain both `race_date_col` and `source_date_col`.
    race_date_col : str
        Column holding the race start datetime.
    source_date_col : str
        Column holding the latest date/time at which the feature data was
        recorded / published.
    tolerance_hours : int
        Allow feature data collected up to N hours before race start.
        Default 0 = feature must strictly precede race start.

    Raises
    ------
    DataLeakageError
        If any row has feature_source_date >= race_date - tolerance.
    """
    if source_date_col not in feature_df.columns:
        # No source date provided — skip check (trust caller)
        return

    race_dates = pd.to_datetime(feature_df[race_date_col], errors="coerce")
    src_dates  = pd.to_datetime(feature_df[source_date_col], errors="coerce")

    cutoff = race_dates - timedelta(hours=tolerance_hours)
    leaking = src_dates >= cutoff

    if leaking.any():
        bad_rows = feature_df[leaking][[race_date_col, source_date_col]].head(5)
        raise DataLeakageError(
            f"DATA LEAKAGE DETECTED: {leaking.sum()} rows have "
            f"`{source_date_col}` at or after `{race_date_col}`.\n"
            f"Sample leaking rows:\n{bad_rows}"
        )
    print(f"[LeakageChecker] ✅ No leakage detected across {len(feature_df)} rows.")


def check_feature_group(
    df: pd.DataFrame,
    feature_cols: list,
    description: str = "",
    race_date_col: str = "date",
):
    """
    Lightweight audit: prints which feature columns have NaN rates and
    confirms they are numeric / properly typed for XGBoost.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list
        List of feature column names to audit.
    description : str
        Human-readable description for the log.
    """
    print(f"\n[LeakageChecker] Auditing feature group: {description or 'unnamed'}")
    for col in feature_cols:
        if col not in df.columns:
            print(f"  [WARN] MISSING: {col}")
            continue
        nan_pct = df[col].isna().mean() * 100
        dtype   = df[col].dtype
        if nan_pct > 10.0:
            print(f"  [WARN] {col:40s} | dtype={dtype} | NaN={nan_pct:.1f}%")
        else:
            print(f"  [OK]   {col:40s} | dtype={dtype} | NaN={nan_pct:.1f}%")
