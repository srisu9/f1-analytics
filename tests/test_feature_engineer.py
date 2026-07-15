import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineer import engineer_features
from src.config import GLOBAL_TOP10_PRIOR, ROLLING_AVG_FILL


@pytest.fixture
def sample_raw_data():
    """Provides a minimal raw dataset mimicking Ergast's merged output."""
    return pd.DataFrame({
        "year": [2020, 2020, 2020, 2020],
        "round": [1, 2, 3, 4],
        "date": ["2020-07-05", "2020-07-12", "2020-07-19", "2020-08-02"],
        "positionOrder": [1, 11, 2, 12],
        "points": [25.0, 0.0, 18.0, 0.0],
        "driverRef": ["leclerc", "leclerc", "leclerc", "leclerc"],
        "constructor_name": ["Ferrari", "Ferrari", "Ferrari", "Ferrari"],
        "circuitRef": ["spielberg", "spielberg", "hungaroring", "silverstone"],
        "dob": ["1997-10-16"] * 4,
        "driver_nationality": ["Monegasque"] * 4,
        "country": ["Austria", "Austria", "Hungary", "UK"],
        "grid": [7, 10, 6, 4],
        "qualifying_position": [7, 10, 6, 4],
        # Required ID / Leakage cols to drop
        "position": ["1", "11", "2", "12"],
        "resultId": [1, 2, 3, 4],
        "time": ["1:30:00"] * 4
    })


def test_engineer_features_drops_leakage(sample_raw_data):
    """Ensure positionOrder and points are dropped to prevent target leakage."""
    df_eng = engineer_features(sample_raw_data)
    assert "positionOrder" not in df_eng.columns
    assert "points" not in df_eng.columns
    assert "time" not in df_eng.columns


def test_engineer_features_creates_top10(sample_raw_data):
    """Ensure the target column Top10 is created correctly."""
    df_eng = engineer_features(sample_raw_data)
    assert "Top10" in df_eng.columns
    # positions: 1, 11, 2, 12 -> 1, 0, 1, 0
    np.testing.assert_array_equal(df_eng["Top10"].values, [1, 0, 1, 0])


def test_driver_top10_rate_prior(sample_raw_data):
    """Ensure new drivers are filled with GLOBAL_TOP10_PRIOR."""
    df_eng = engineer_features(sample_raw_data)
    
    # First race: no prior history, should be GLOBAL_TOP10_PRIOR
    assert df_eng.loc[0, "driver_top10_rate"] == GLOBAL_TOP10_PRIOR
    
    # Second race: 1 prior race, 1 Top 10 finish (100% rate)
    assert df_eng.loc[1, "driver_top10_rate"] == 1.0
    
    # Third race: 2 prior races, 1 Top 10 finish (50% rate)
    assert df_eng.loc[2, "driver_top10_rate"] == 0.50
    
    # Fourth race: 3 prior races, 2 Top 10 finishes (66.6% rate)
    assert np.isclose(df_eng.loc[3, "driver_top10_rate"], 2/3)


def test_rolling_average_finish(sample_raw_data):
    """Ensure rolling_avg_finish uses .shift(1) and fills with ROLLING_AVG_FILL."""
    df_eng = engineer_features(sample_raw_data)
    
    # First race: filled with default
    assert df_eng.loc[0, "rolling_avg_finish_3"] == ROLLING_AVG_FILL
    
    # Second race: average of race 1
    assert df_eng.loc[1, "rolling_avg_finish_3"] == 1.0
    
    # Third race: average of race 1 and 2 (1 and 11) -> 6.0
    assert df_eng.loc[2, "rolling_avg_finish_3"] == 6.0
