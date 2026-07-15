"""
v4_phase4_fastf1.py
===================
Downloads and builds FastF1 telemetry aggregates (avg speed, top speed,
throttle%, brake%, sector std dev, and tire degradation estimates)
for the 2023 test season. Trains an XGBoost model, and evaluates
metrics vs baseline.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data_loader import load_raw_data, merge_datasets
from src.feature_engineer import engineer_features
from src.features.phase4_telemetry import build_phase4_features, add_phase4_to_df, PHASE4_FEATURES
from notebooks.v4_training_pipeline import walk_forward_evaluate, V3_NUMERICAL, V3_CATEGORICAL

# 1. Load Ergast base data
print("[V4 Phase 4] Loading Ergast base datasets...")
DATA_DIR = os.path.join(ROOT, "data", "raw")
dfs = load_raw_data(DATA_DIR)
merged = merge_datasets(dfs)
df_eng = engineer_features(merged.copy())

# Ensure key columns exist
for col in ["year", "round", "circuitRef", "driverRef", "constructor_name"]:
    if col not in df_eng.columns and col in merged.columns:
        df_eng[col] = merged[col].values
df_eng = df_eng.dropna(subset=["Top10"])

# 2. Define races map
sim_races = [
    (2023, 1, "Bahrain"),
    (2023, 2, "Saudi Arabia"),
    (2023, 3, "Australia"),
    (2023, 4, "Azerbaijan"),
    (2023, 5, "Miami"),
    (2023, 6, "Monaco"),
    (2023, 7, "Spain"),
    (2023, 8, "Canada"),
    (2023, 9, "Austria"),
    (2023, 10, "Great Britain"),
]

gp_map = {(year, rnd): gp for year, rnd, gp in sim_races}
years = sorted(list(set(y for y, _, _ in sim_races)))

print(f"\n[V4 Phase 4] Fetching FastF1 telemetry aggregates for {len(sim_races)} races...")

try:
    phase4_df = build_phase4_features(gp_map)
except Exception as e:
    print(f"[V4 Phase 4] Warning: Telemetry download failed: {e}.")
    phase4_df = pd.DataFrame()

# 3. Add Phase 4 features to the dataset
df_p4 = add_phase4_to_df(df_eng.copy(), phase4_df)

# Evaluate on the 2023 test season
print("\n[V4 Phase 4] Evaluating metrics on the 2023 Season fold...")

df_train = df_p4[df_p4["year"] < 2023].copy()
df_test = df_p4[df_p4["year"] == 2023].copy()

global_mean = df_train["Top10"].mean()
for cat in V3_CATEGORICAL:
    stats = df_train.groupby(cat)["Top10"].agg(["mean", "count"])
    smooth_vals = (stats["count"] * stats["mean"] + 10 * global_mean) / (stats["count"] + 10)
    smooth_map = smooth_vals.to_dict()
    df_train[cat + "_enc"] = df_train[cat].map(smooth_map).fillna(global_mean)
    df_test[cat + "_enc"] = df_test[cat].map(smooth_map).fillna(global_mean)

# Impute NaN
for col in V3_NUMERICAL + PHASE4_FEATURES:
    if col in df_train.columns:
        med = df_train[col].median()
        df_train[col] = df_train[col].fillna(med)
        df_test[col] = df_test[col].fillna(med)

# Feature sets
num_v3 = V3_NUMERICAL + [c + "_enc" for c in V3_CATEGORICAL]
num_v4 = V3_NUMERICAL + [col for col in PHASE4_FEATURES if col in df_p4.columns] + [c + "_enc" for c in V3_CATEGORICAL]

X_train_v3, y_train = df_train[num_v3], df_train["Top10"]
X_test_v3, y_test = df_test[num_v3], df_test["Top10"]

X_train_v4 = df_train[num_v4]
X_test_v4 = df_test[num_v4]

# Model V3
model_v3 = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
model_v3.fit(X_train_v3, y_train)
preds_v3 = model_v3.predict(X_test_v3)
acc_v3 = accuracy_score(y_test, preds_v3)

# Model V4
model_v4 = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
model_v4.fit(X_train_v4, y_train)
preds_v4 = model_v4.predict(X_test_v4)
acc_v4 = accuracy_score(y_test, preds_v4)

print(f"\nResults for 2023 Season fold:")
print(f"  V3 Baseline Accuracy  : {acc_v3:.4f}")
print(f"  V4 Phase 4 Accuracy   : {acc_v4:.4f}")
print(f"  Delta Accuracy        : {acc_v4 - acc_v3:+.4f}")
