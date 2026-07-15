"""
v4_training_pipeline.py
========================
Central training pipeline for F1 Analytics Version 4.

Runs all 5 phases sequentially:
  0. Baseline (V3 features only)
  2. Phase 2: Circuit history
  3. Phase 3: Weather + historical safety car rate
  5. Phase 5: Championship & rolling form

For each phase:
  - Trains XGBoost with walk-forward validation (2019–2024)
  - Evaluates: Accuracy, Precision, Recall, F1, ROC-AUC
  - Runs SHAP analysis
  - Saves model to models/v4_phase{N}_xgb.joblib
  - Saves metrics to reports/v4_phase{N}_metrics.json
  - Compares vs previous phase

Usage:
    python notebooks/v4_training_pipeline.py

Or run per-phase with --phase flag:
    python notebooks/v4_training_pipeline.py --phase 0 2
"""
import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.config import (
    PATHS,
    TRAIN_YEARS,
    TEST_YEARS,
    TARGET_COL,
    XGB_PARAMS,
    TARGET_ENCODE_SMOOTHING,
    V3_NUMERICAL,
    V3_CATEGORICAL
)
from src.data_loader import load_raw_data, merge_datasets
from src.feature_engineer import engineer_features
from src.leakage_checker import check_feature_group
from src.shap_reporter import generate_shap_report

from src.features.phase2_circuit_history import add_phase2_features, PHASE2_FEATURES
from src.features.phase3_weather_safety import add_phase3_features, PHASE3_FEATURES
from src.features.phase5_championship import add_phase5_features, PHASE5_FEATURES

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────
os.makedirs(PATHS["models_dir"], exist_ok=True)
os.makedirs(PATHS["reports_dir"], exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def target_encode(X_train, X_test, y_train, categorical_cols, smoothing=TARGET_ENCODE_SMOOTHING):
    """Smoothed target encoding — fit on train only, apply to test."""
    global_mean = y_train.mean()
    encoders = {}
    for cat in categorical_cols:
        stats = y_train.groupby(X_train[cat]).agg(["mean", "count"])
        smooth_vals = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )
        smooth_map = smooth_vals.to_dict()
        X_train[cat + "_encoded"] = X_train[cat].map(smooth_map).fillna(global_mean)
        if X_test is not None and not X_test.empty:
            X_test[cat  + "_encoded"] = X_test[cat].map(smooth_map).fillna(global_mean)
        encoders[cat] = {"map": smooth_map, "global_mean": global_mean}
    X_train = X_train.drop(columns=categorical_cols)
    if X_test is not None and not X_test.empty:
        X_test  = X_test.drop(columns=categorical_cols)
    return X_train, X_test, encoders


def walk_forward_evaluate(df, feature_cols, categorical_cols, label):
    """
    Walk-forward validation: train on years < test_year, test on test_year.
    Averages metrics over all test years.
    Returns: avg_metrics, final_model, final_X_train, final_preprocessor_dict
    """
    all_metrics = []
    all_y_true, all_y_prob = [], []

    for test_year in TEST_YEARS:
        train_years = [y for y in df["year"].unique() if y < test_year]
        if not train_years:
            continue

        df_train = df[df["year"].isin(train_years)].copy()
        df_test  = df[df["year"] == test_year].copy()

        if df_train.empty or df_test.empty:
            continue

        all_cols = feature_cols + categorical_cols
        available = [c for c in all_cols if c in df_train.columns and c in df_test.columns]

        X_tr = df_train[available].copy()
        y_tr = df_train[TARGET_COL].copy()
        X_te = df_test[available].copy()
        y_te = df_test[TARGET_COL].copy()

        # Impute NaN with column median (from training set)
        for col in X_tr.select_dtypes(include=[np.number]).columns:
            med = X_tr[col].median()
            X_tr[col] = X_tr[col].fillna(med)
            X_te[col] = X_te[col].fillna(med)

        cat_in_available = [c for c in categorical_cols if c in available]
        X_tr, X_te, _ = target_encode(X_tr, X_te, y_tr, cat_in_available, smoothing=TARGET_ENCODE_SMOOTHING)

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, verbose=False)

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        all_y_true.extend(y_te.tolist())
        all_y_prob.extend(y_prob.tolist())

        m = {
            "year":      test_year,
            "accuracy":  accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall":    recall_score(y_te, y_pred, zero_division=0),
            "f1":        f1_score(y_te, y_pred, zero_division=0),
            "auc":       roc_auc_score(y_te, y_prob),
        }
        all_metrics.append(m)

    if not all_metrics:
        return {}, None, None, None

    avg = {
        "accuracy":  np.mean([m["accuracy"]  for m in all_metrics]),
        "precision": np.mean([m["precision"] for m in all_metrics]),
        "recall":    np.mean([m["recall"]    for m in all_metrics]),
        "f1":        np.mean([m["f1"]        for m in all_metrics]),
        "auc":       np.mean([m["auc"]       for m in all_metrics]),
        "per_year":  all_metrics,
    }

    # ── Retrain on all train data for final model ─────────────────────────────
    df_train_full = df[df["year"] <= max(TEST_YEARS)].copy()
    
    all_cols = feature_cols + categorical_cols
    available = [c for c in all_cols if c in df_train_full.columns]

    X_tr_f = df_train_full[available].copy()
    y_tr_f = df_train_full[TARGET_COL].copy()

    imputation_medians = {}
    for col in X_tr_f.select_dtypes(include=[np.number]).columns:
        med = float(X_tr_f[col].median()) # ensure native float for json serialization
        imputation_medians[col] = med if not pd.isna(med) else 0.0
        X_tr_f[col] = X_tr_f[col].fillna(imputation_medians[col])

    cat_in = [c for c in categorical_cols if c in available]
    X_tr_f, _, final_encoders = target_encode(X_tr_f, None, y_tr_f, cat_in, smoothing=TARGET_ENCODE_SMOOTHING)

    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X_tr_f, y_tr_f, verbose=False)

    final_preprocessor = {
        "encoders": final_encoders,
        "features": list(X_tr_f.columns),
        "imputation_medians": imputation_medians,
        "smoothing": TARGET_ENCODE_SMOOTHING
    }

    return avg, final_model, X_tr_f, final_preprocessor


def print_comparison(label, metrics, baseline=None):
    """Pretty-prints metrics and delta vs baseline."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for k in ["accuracy", "precision", "recall", "f1", "auc"]:
        v = metrics.get(k, 0)
        if baseline:
            delta = v - baseline.get(k, 0)
            sign = "+" if delta > 0 else ("-" if delta < 0 else "=")
            print(f"  {k:12s}: {v:.4f}   {sign} {abs(delta):.4f}")
        else:
            print(f"  {k:12s}: {v:.4f}")


# ────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────────

def main(run_phases=None):
    print("\n[V4 Pipeline] Loading Ergast data...")
    dfs = load_raw_data(PATHS["data_raw"])
    merged = merge_datasets(dfs)

    # engineer_features needs positionOrder (raw) to compute rolling stats,
    # then drops it internally. Pass the raw merged df, not cleaned.
    df_eng = engineer_features(merged.copy())

    # Keep a cleaned reference (positionOrder dropped) for Phase 2/5 stat
    # computations that need statusId etc. from the original merged frame.
    cleaned = merged.copy()   # raw merged — Phase 2/5 use positionOrder from here

    # Ensure key join columns survive into df_eng
    for col in ["year", "round", "circuitRef", "driverRef", "constructor_name"]:
        if col not in df_eng.columns and col in merged.columns:
            df_eng[col] = merged[col].values

    df_eng = df_eng.dropna(subset=[TARGET_COL])

    # ── BASELINE (V3) ─────────────────────────────────────────────────────
    if run_phases is None or 0 in run_phases:
        print("\n[V4 Pipeline] --- Phase 0: V3 BASELINE ---")
        check_feature_group(df_eng, V3_NUMERICAL, "V3 Baseline Numerical")
        metrics_v3, model_v3, X_v3, preprocessor_v3 = walk_forward_evaluate(
            df_eng, V3_NUMERICAL, V3_CATEGORICAL, "V3 Baseline"
        )
        print_comparison("V3 BASELINE", metrics_v3)
        metrics_v3["phase"] = "v3_baseline"

        with open(PATHS["metrics_v3"], "w") as f:
            json.dump(metrics_v3, f, indent=2)

        if model_v3 is not None and X_v3 is not None:
            # Baseline is not the final model, just an intermediate artifact
            pass
            
        baseline = metrics_v3
    else:
        baseline = json.load(open(PATHS["metrics_v3"])) if os.path.exists(PATHS["metrics_v3"]) else {}

    # ── PHASE 2: Circuit History (Ergast-only, no FastF1 needed) ──────────
    if run_phases is None or 2 in run_phases:
        print("\n[V4 Pipeline] --- Phase 2: CIRCUIT HISTORY ---")
        df_p2 = df_eng.copy()
        df_p2 = add_phase2_features(df_p2, cleaned)
        check_feature_group(df_p2, PHASE2_FEATURES, "Phase 2 Circuit History")

        feats_p2 = V3_NUMERICAL + PHASE2_FEATURES
        metrics_p2, model_p2, X_p2, preprocessor_p2 = walk_forward_evaluate(
            df_p2, feats_p2, V3_CATEGORICAL, "Phase 2"
        )
        print_comparison("PHASE 2: Circuit History", metrics_p2, baseline)
        metrics_p2["phase"] = "v4_phase2"

        with open(PATHS["metrics_phase2"], "w") as f:
            json.dump(metrics_p2, f, indent=2)

        if model_p2 is not None and X_p2 is not None:
            joblib.dump(model_p2, PATHS["model_phase2"])
            # Save Phase 2 as the main V4 model (best overall performer)
            joblib.dump(model_p2, PATHS["model_final"])
            print(f"\n[V4 Pipeline] [OK] Champion V4 model saved -> {PATHS['model_final']}")
            
            # Save the engineered feature dataset for Streamlit integration
            df_p2.to_csv(PATHS["model_ready_csv"], index=False)
            print(f"[V4 Pipeline] [OK] Saved V4 engineered features to {PATHS['model_ready_csv']}")

            # Save the preprocessor metadata containing the V4 features list, medians, encoders
            joblib.dump(preprocessor_p2, PATHS["preprocessor"])
            print(f"[V4 Pipeline] [OK] Saved V4 preprocessor map -> {PATHS['preprocessor']}")
            
            generate_shap_report(
                model_p2, X_p2, list(X_p2.columns),
                PATHS["reports_dir"], phase_label="v4_phase2"
            )
        baseline = metrics_p2

    # ── PHASE 5: Championship & Rolling Form (Ergast-only) ────────────────
    if run_phases is None or 5 in run_phases:
        print("\n[V4 Pipeline] --- Phase 5: CHAMPIONSHIP & ROLLING FORM ---")
        df_p5 = df_eng.copy()

        # Re-apply phase 2 if it was run
        if os.path.exists(PATHS["metrics_phase2"]):
            df_p5 = add_phase2_features(df_p5, cleaned)

        # Load standings CSVs
        driver_standings = pd.read_csv(
            os.path.join(PATHS["data_raw"], "driver_standings.csv"), na_values=["\\N"]
        )
        constructor_standings = pd.read_csv(
            os.path.join(PATHS["data_raw"], "constructor_standings.csv"), na_values=["\\N"]
        )

        df_p5 = add_phase5_features(df_p5, driver_standings, constructor_standings, cleaned)
        check_feature_group(df_p5, PHASE5_FEATURES, "Phase 5 Championship")

        prev_feats = V3_NUMERICAL + PHASE2_FEATURES if os.path.exists(PATHS["metrics_phase2"]) else V3_NUMERICAL
        feats_p5 = prev_feats + PHASE5_FEATURES
        metrics_p5, model_p5, X_p5, preprocessor_p5 = walk_forward_evaluate(
            df_p5, feats_p5, V3_CATEGORICAL, "Phase 5"
        )
        print_comparison("PHASE 5: Championship & Rolling Form", metrics_p5, baseline)
        metrics_p5["phase"] = "v4_phase5"

        with open(PATHS["metrics_phase5"], "w") as f:
            json.dump(metrics_p5, f, indent=2)

        if model_p5 is not None and X_p5 is not None:
            joblib.dump(model_p5, PATHS["model_phase5"])
            generate_shap_report(
                model_p5, X_p5, list(X_p5.columns),
                PATHS["reports_dir"], phase_label="v4_phase5"
            )

    # ── PHASE 3: Weather + Safety Car (requires FastF1 for weather) ───────
    if run_phases is None or 3 in run_phases:
        print("\n[V4 Pipeline] --- Phase 3: WEATHER + SAFETY CAR ---")
        df_p3 = df_eng.copy()
        df_p3 = add_phase2_features(df_p3, cleaned)

        driver_standings = pd.read_csv(
            os.path.join(PATHS["data_raw"], "driver_standings.csv"), na_values=["\\N"]
        )
        constructor_standings = pd.read_csv(
            os.path.join(PATHS["data_raw"], "constructor_standings.csv"), na_values=["\\N"]
        )
        df_p3 = add_phase5_features(df_p3, driver_standings, constructor_standings, cleaned)

        races_df = dfs["races"].copy()
        races_df.rename(columns={"name": "race_name", "time": "race_time", "url": "race_url"}, inplace=True)

        df_p3 = add_phase3_features(
            df_p3,
            ergast_races_df=races_df,
            ergast_results_df=cleaned,
            year_gp_map={},   # Empty — skip FastF1 weather for now
        )
        check_feature_group(df_p3, PHASE3_FEATURES, "Phase 3 Weather & Safety Car")

        feats_p3 = V3_NUMERICAL + PHASE2_FEATURES + PHASE5_FEATURES + PHASE3_FEATURES
        metrics_p3, model_p3, X_p3, preprocessor_p3 = walk_forward_evaluate(
            df_p3, feats_p3, V3_CATEGORICAL, "Phase 3"
        )
        print_comparison("PHASE 3: Weather + Safety Car", metrics_p3, baseline)
        metrics_p3["phase"] = "v4_phase3"

        with open(PATHS["metrics_phase3"], "w") as f:
            json.dump(metrics_p3, f, indent=2)

        if model_p3 is not None and X_p3 is not None:
            pass

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  V4 TRAINING PIPELINE COMPLETE")
    print("="*60)
    print("  Metrics saved -> reports/")
    print("  Models saved  -> models/")
    print("  SHAP plots    -> reports/*.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4 Training Pipeline")
    parser.add_argument("--phase", type=int, nargs="+",
                        help="Which phases to run (0=baseline, 2, 3, 5). Default: all.")
    args = parser.parse_args()
    main(run_phases=args.phase)
