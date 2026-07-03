import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os


def prepare_ml_data(df, target_col="Top10", split_year=2019):
    """
    Splits the data temporally.
    Performs target encoding on high-cardinality categorical features
    using training data statistics only to avoid data leakage.
    """
    numerical_features = [
        "grid", "driver_age", "driver_experience", "driver_win_rate",
        "constructor_win_rate", "rolling_avg_finish_3", "rolling_avg_finish_5",
        "prev_race_finish", "home_race", "grid_qualifying_diff",
        "constructor_season_points", "lat", "lng", "alt"
    ]

    categorical_features = ["driverRef", "constructor_name", "circuitRef"]
    all_features = numerical_features + categorical_features

    train_mask = df["year"] < split_year
    test_mask  = df["year"] >= split_year

    df_train = df[train_mask].copy()
    df_test  = df[test_mask].copy()

    X_train = df_train[all_features].copy()
    y_train = df_train[target_col].copy()
    X_test  = df_test[all_features].copy()
    y_test  = df_test[target_col].copy()

    global_mean = y_train.mean()
    smoothing   = 10

    encoders = {}
    for cat in categorical_features:
        stats = y_train.groupby(X_train[cat]).agg(["mean", "count"])
        smooth_vals = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        smooth_map  = smooth_vals.to_dict()

        X_train[cat + "_encoded"] = X_train[cat].map(smooth_map).fillna(global_mean)
        X_test[cat  + "_encoded"] = X_test[cat].map(smooth_map).fillna(global_mean)

        encoders[cat] = {"map": smooth_map, "global_mean": global_mean}

    X_train.drop(columns=categorical_features, inplace=True)
    X_test.drop(columns=categorical_features, inplace=True)

    scaler = StandardScaler()
    cols_to_scale = numerical_features + [c + "_encoded" for c in categorical_features]

    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

    preprocessor = {
        "scaler": scaler,
        "encoders": encoders,
        "features": cols_to_scale,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features
    }

    return X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, preprocessor


def prepare_fold_data(df, train_years, test_year, target_col="Top10"):
    """
    Builds a single walk-forward fold:
      - train on all rows where year is in train_years
      - test  on all rows where year == test_year
    Returns X_train, y_train, X_test, y_test (unscaled), and the encoder map.
    """
    numerical_features = [
        "grid", "driver_age", "driver_experience", "driver_win_rate",
        "constructor_win_rate", "rolling_avg_finish_3", "rolling_avg_finish_5",
        "prev_race_finish", "home_race", "grid_qualifying_diff",
        "constructor_season_points", "lat", "lng", "alt"
    ]
    categorical_features = ["driverRef", "constructor_name", "circuitRef"]
    all_features = numerical_features + categorical_features

    df_train = df[df["year"].isin(train_years)].copy()
    df_test  = df[df["year"] == test_year].copy()

    if df_train.empty or df_test.empty:
        return None

    X_train = df_train[all_features].copy()
    y_train = df_train[target_col].copy()
    X_test  = df_test[all_features].copy()
    y_test  = df_test[target_col].copy()

    global_mean = y_train.mean()
    smoothing   = 10

    encoders = {}
    for cat in categorical_features:
        stats = y_train.groupby(X_train[cat]).agg(["mean", "count"])
        smooth_vals = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        smooth_map  = smooth_vals.to_dict()

        X_train[cat + "_encoded"] = X_train[cat].map(smooth_map).fillna(global_mean)
        X_test[cat  + "_encoded"] = X_test[cat].map(smooth_map).fillna(global_mean)
        encoders[cat] = {"map": smooth_map, "global_mean": global_mean}

    X_train.drop(columns=categorical_features, inplace=True)
    X_test.drop(columns=categorical_features, inplace=True)

    return X_train, y_train, X_test, y_test, encoders


def evaluate_predictions(y_true, y_pred, y_prob):
    """
    Returns a dictionary of evaluation metrics.
    """
    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision":        precision_score(y_true, y_pred, zero_division=0),
        "recall":           recall_score(y_true, y_pred, zero_division=0),
        "f1":               f1_score(y_true, y_pred, zero_division=0),
        "auc":              roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
