"""
test_phase2_circuit_history.py
================================
Tests for Phase 2 Bayesian-smoothed circuit history features.
"""
import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.phase2_circuit_history import add_phase2_features


@pytest.fixture
def raw_data():
    """Minimal raw merged dataset with 4 races at two circuits."""
    return pd.DataFrame({
        "year":          [2020, 2020, 2021, 2021],
        "round":         [1,    2,    1,    2   ],
        "date":          ["2020-07-05", "2020-07-12", "2021-03-28", "2021-04-18"],
        "positionOrder": [1,    10,   3,    15  ],
        "grid":          [3,    6,    2,    8   ],
        "driverRef":     ["hamilton", "hamilton", "hamilton", "hamilton"],
        "circuitRef":    ["bahrain",  "silverstone", "bahrain", "imola"],
        # Extra cols to simulate merged df
        "constructor_name": ["Mercedes"] * 4,
        "race_name":        ["Bahrain GP", "British GP", "Bahrain GP", "Emilia GP"],
    })


@pytest.fixture
def base_df(raw_data):
    """Minimal base dataframe (after engineer_features) that we merge Phase 2 into."""
    return pd.DataFrame({
        "year":          raw_data["year"].values,
        "round":         raw_data["round"].values,
        "date":          pd.to_datetime(raw_data["date"]),
        "driverRef":     raw_data["driverRef"].values,
        "circuitRef":    raw_data["circuitRef"].values,
        "constructor_name": raw_data["constructor_name"].values,
    })


def test_smoothed_circuit_avg_finish_no_nan(base_df, raw_data):
    """Ensure smoothed_circuit_avg_finish has no NaN values after join."""
    df_out = add_phase2_features(base_df, raw_data, K=4)
    assert df_out["smoothed_circuit_avg_finish"].isna().sum() == 0


def test_first_visit_uses_career_avg(base_df, raw_data):
    """
    First visit to a circuit should be fully determined by career prior (K smoothing).
    With 0 visits and K=4: smoothed = (0*circuit_avg + 4*career_avg) / (0+4) = career_avg
    """
    df_out = add_phase2_features(base_df, raw_data, K=4)
    
    # Row 0: Hamilton's first ever race at Bahrain (and first ever race).
    # career_avg for row 0 = 12.0 (default, no prior races).
    # n_visits = 0 → smoothed = (0 + 4*12.0) / (0+4) = 12.0
    row0 = df_out.iloc[0]
    assert np.isclose(row0["smoothed_circuit_avg_finish"], 12.0)


def test_circuit_grid_finish_corr_no_nan(base_df, raw_data):
    """Ensure circuit_grid_finish_corr has no NaN values."""
    df_out = add_phase2_features(base_df, raw_data, K=4)
    assert df_out["circuit_grid_finish_corr"].isna().sum() == 0
