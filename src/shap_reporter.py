"""
shap_reporter.py
================
Generates SHAP feature importance reports for V4 model evaluation.

Produces:
  - Beeswarm summary plot  (global feature importance)
  - Bar plot               (mean |SHAP value| per feature)
  - JSON report            (mean |SHAP| per feature, for programmatic comparison)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    _shap_available = True
except ImportError:
    _shap_available = False
    print("[SHAPReporter] shap not installed.")


def generate_shap_report(
    model,
    X: pd.DataFrame,
    feature_names: list,
    output_dir: str,
    phase_label: str = "v4",
    max_display: int = 20,
) -> dict:
    """
    Generates SHAP beeswarm and bar plots, saves to output_dir.
    Returns dict of mean |SHAP| per feature (for comparison table).

    Parameters
    ----------
    model       : fitted XGBoost / tree model
    X           : feature matrix (DataFrame)
    feature_names : list of column names
    output_dir  : directory to save plots and JSON
    phase_label : prefix for output files, e.g. 'v3_baseline' or 'v4_phase1'
    max_display : max features shown in plot
    """
    os.makedirs(output_dir, exist_ok=True)

    if not _shap_available:
        print("[SHAPReporter] SHAP not available — skipping.")
        return {}

    try:
        explainer = shap.TreeExplainer(model)
        # Use a sample for speed if X is large
        sample = X if len(X) <= 2000 else X.sample(2000, random_state=42)
        shap_values = explainer(sample)

        # Handle multi-output (some models return 3D)
        sv = shap_values
        if hasattr(sv, "values") and sv.values.ndim == 3:
            sv = shap.Explanation(
                values=sv.values[:, :, 1],
                base_values=sv.base_values[:, 1] if sv.base_values.ndim > 1 else sv.base_values,
                data=sv.data,
                feature_names=feature_names,
            )

        # --- Beeswarm plot ---
        fig_bee, _ = plt.subplots(figsize=(10, 0.4 * min(max_display, len(feature_names)) + 2))
        fig_bee.patch.set_facecolor("#0e0f12")
        shap.plots.beeswarm(sv, max_display=max_display, show=False)
        plt.tight_layout()
        bee_path = os.path.join(output_dir, f"{phase_label}_shap_beeswarm.png")
        plt.savefig(bee_path, dpi=120, bbox_inches="tight", facecolor="#0e0f12")
        plt.close(fig_bee)
        print(f"[SHAPReporter] Saved beeswarm → {bee_path}")

        # --- Bar plot ---
        fig_bar, _ = plt.subplots(figsize=(10, 0.4 * min(max_display, len(feature_names)) + 2))
        fig_bar.patch.set_facecolor("#0e0f12")
        shap.plots.bar(sv, max_display=max_display, show=False)
        plt.tight_layout()
        bar_path = os.path.join(output_dir, f"{phase_label}_shap_bar.png")
        plt.savefig(bar_path, dpi=120, bbox_inches="tight", facecolor="#0e0f12")
        plt.close(fig_bar)
        print(f"[SHAPReporter] Saved bar plot → {bar_path}")

        # --- JSON importance ---
        mean_abs_shap = np.abs(sv.values).mean(axis=0)
        importance = dict(zip(feature_names, mean_abs_shap.tolist()))
        importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        json_path = os.path.join(output_dir, f"{phase_label}_shap_importance.json")
        with open(json_path, "w") as f:
            json.dump(importance_sorted, f, indent=2)
        print(f"[SHAPReporter] Saved importance JSON → {json_path}")

        return importance_sorted

    except Exception as exc:
        print(f"[SHAPReporter] Error generating report: {exc}")
        return {}


def compare_shap_reports(report_paths: dict) -> pd.DataFrame:
    """
    Loads multiple SHAP JSON reports and builds a comparison DataFrame.

    Parameters
    ----------
    report_paths : dict
        {phase_label: path_to_shap_importance.json}

    Returns
    -------
    DataFrame with features as rows and phases as columns.
    """
    data = {}
    for label, path in report_paths.items():
        if os.path.exists(path):
            with open(path) as f:
                data[label] = json.load(f)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).fillna(0)
    df = df.sort_values(list(data.keys())[-1], ascending=False)
    return df
