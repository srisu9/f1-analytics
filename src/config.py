"""
config.py
=========
Central configuration for the F1 Analytics V4 project.

All file paths, model constants, and feature definitions live here.
Import this module anywhere in the project to avoid hardcoded paths.

Usage:
    from src.config import PATHS, MODEL_FEATURES, XGB_PARAMS
"""

import os

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Version ───────────────────────────────────────────────────────────────────
PROJECT_VERSION = "4.0.0"
PROJECT_NAME    = "F1 Analytics AI Platform"

# ── Paths ─────────────────────────────────────────────────────────────────────
PATHS = {
    # Data
    "data_raw":          os.path.join(ROOT, "data", "raw"),
    "data_processed":    os.path.join(ROOT, "data", "processed"),
    "model_ready_csv":   os.path.join(ROOT, "data", "processed", "model_ready.csv"),
    "fastf1_cache":      os.path.join(ROOT, "data", "fastf1"),

    # Models
    "models_dir":        os.path.join(ROOT, "models"),
    "model_final":       os.path.join(ROOT, "models", "v4_xgb_final.joblib"),
    "model_phase2":      os.path.join(ROOT, "models", "v4_phase2_xgb.joblib"),
    "model_phase5":      os.path.join(ROOT, "models", "v4_phase5_xgb.joblib"),
    "preprocessor":      os.path.join(ROOT, "models", "preprocessor.joblib"),

    # Reports
    "reports_dir":       os.path.join(ROOT, "reports"),
    "metrics_v3":        os.path.join(ROOT, "reports", "v3_baseline_metrics.json"),
    "metrics_phase2":    os.path.join(ROOT, "reports", "v4_phase2_metrics.json"),
    "metrics_phase3":    os.path.join(ROOT, "reports", "v4_phase3_metrics.json"),
    "metrics_phase5":    os.path.join(ROOT, "reports", "v4_phase5_metrics.json"),
}

# ── Training Configuration ────────────────────────────────────────────────────
TRAIN_YEARS = list(range(1994, 2019))
TEST_YEARS  = list(range(2019, 2025))
TARGET_COL  = "Top10"

# ── XGBoost Hyperparameters ───────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "eval_metric":       "logloss",
    "random_state":      42,
    "n_jobs":            -1,
}

# ── Target Encoding ───────────────────────────────────────────────────────────
TARGET_ENCODE_SMOOTHING = 10   # Bayesian smoothing factor (k)
TARGET_ENCODE_COLS = ["driverRef", "constructor_name", "circuitRef"]

# ── Baseline Feature Columns (V3) ─────────────────────────────────────────────
# NOTE: driver_top10_rate and constructor_top10_rate were previously named
# driver_win_rate and constructor_win_rate.  The rename happened in V4.0.0.
V3_NUMERICAL = [
    "grid",
    "driver_age",
    "driver_experience",
    "driver_top10_rate",        # previously: driver_win_rate
    "constructor_top10_rate",   # previously: constructor_win_rate
    "rolling_avg_finish_3",
    "rolling_avg_finish_5",
    "prev_race_finish",
    "home_race",
    "grid_qualifying_diff",
    "constructor_season_points",
    "lat",
    "lng",
    "alt",
]
V3_CATEGORICAL = ["driverRef", "constructor_name", "circuitRef"]

# ── Global Top-10 Rate Prior ──────────────────────────────────────────────────
# With 20 drivers and 10 point-scoring positions, the base rate is exactly 0.5.
# Used as fillna default for new drivers/constructors with no history.
GLOBAL_TOP10_PRIOR = 0.50

# ── Rolling Average Fill Default ──────────────────────────────────────────────
# Mid-field default for drivers/constructors with no prior races.
ROLLING_AVG_FILL = 12.0

# ── Circuit Overtaking Index (used in H2H tab) ────────────────────────────────
# Approximate historical grid-to-finish correlation per circuit.
# Higher = grid position matters more (harder to overtake).
CIRCUIT_OVERTAKING_INDEX = {
    "monaco":       0.88,
    "singapore":    0.82,
    "jeddah":       0.75,
    "baku":         0.70,
    "hungaroring":  0.65,
    "silverstone":  0.55,
    "bahrain":      0.58,
    "monza":        0.52,
    "interlagos":   0.48,
    "red_bull_ring": 0.48,
    "spa":          0.45,
    "albert_park":  0.50,
    "suzuka":       0.53,
    "americas":     0.50,
    "zandvoort":    0.58,
    "yas_marina":   0.52,
    "losail":       0.45,
    "miami":        0.55,
    "las_vegas":    0.50,
    "rodriguez":    0.50,
    "imola":        0.60,
    "shanghai":     0.52,
}
CIRCUIT_OVERTAKING_DEFAULT = 0.50  # neutral fallback

# ── Weather Simulation Multipliers ───────────────────────────────────────────
# Adjusts features that are legitimately influenced by wet conditions.
# IMPORTANT: Only form/pace features are adjusted. Immutable stats (career
# starts, experience counts) are NEVER modified.
WEATHER_SIM = {
    "Dry":   {},  # no adjustment
    "Mixed": {
        "rolling_avg_finish_3":  +1.5,   # pace uncertainty additive
        "rolling_avg_finish_5":  +1.0,
        "driver_top10_rate":     0.97,   # slight reliability penalty (multiplicative)
        "constructor_top10_rate": 0.97,
    },
    "Wet": {
        "rolling_avg_finish_3":  +3.0,   # higher chaos → worse average expected
        "rolling_avg_finish_5":  +2.0,
        "driver_top10_rate":     0.90,   # wet reliability penalty
        "constructor_top10_rate": 0.88,
        "grid_qualifying_diff":  -1.0,   # grid advantage reduced in wet
    },
}
