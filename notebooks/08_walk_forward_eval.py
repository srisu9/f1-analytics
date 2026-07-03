\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
   

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

                                             
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: catboost not installed. Skipping CatBoost. Install with: pip install catboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: lightgbm not installed. Skipping LightGBM. Install with: pip install lightgbm")

from src.model_trainer import prepare_fold_data

                                                
DATA_PATH   = "data/processed/model_ready.csv"
MODELS_DIR  = "models"
FIGURES_DIR = "reports/figures"
TEST_YEARS  = list(range(2019, 2026))              

                                                            
ENS_WEIGHTS = {"xgb": 0.4, "cat": 0.4, "lgb": 0.2}

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def build_xgb():
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )


def build_cat():
    if not HAS_CATBOOST:
        return None
    return CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=0
    )


def build_lgb():
    if not HAS_LIGHTGBM:
        return None
    return LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )


def eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_test, y_prob),
    }


def ensemble_proba(models_dict, weights, X_test):
    """Weighted average of predict_proba outputs."""
    active = {k: m for k, m in models_dict.items() if m is not None}
    total_w = sum(weights[k] for k in active)
    blend = np.zeros(len(X_test))
    for key, model in active.items():
        blend += (weights[key] / total_w) * model.predict_proba(X_test)[:, 1]
    return blend


def main():
    print("=" * 60)
    print("F1 Analytics — Walk-Forward Cross-Validation")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape[0]:,} rows, {df.shape[1]} columns")

    all_years = sorted(df["year"].unique())
    print(f"Years in dataset: {all_years[0]} – {all_years[-1]}\n")

                                  
    fold_records = []

    for test_year in TEST_YEARS:
        train_years = [y for y in all_years if y < test_year]
        if len(train_years) < 3:
            print(f"Skipping {test_year}: not enough training years.")
            continue

        result = prepare_fold_data(df, train_years, test_year)
        if result is None:
            print(f"Skipping {test_year}: empty train or test split.")
            continue

        X_train, y_train, X_test, y_test, _ = result
        print(f"\n[Fold {test_year}]  Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")

        fold_models = {}

                                                       
        xgb = build_xgb()
        xgb.fit(X_train, y_train)
        fold_models["xgb"] = xgb
        m = eval_model(xgb, X_test, y_test)
        print(f"  XGBoost  -> Acc: {m['accuracy']:.4f}  F1: {m['f1']:.4f}  AUC: {m['auc']:.4f}")
        fold_records.append({"model": "XGBoost", "year": test_year, **m})

                                                       
        cat = build_cat()
        if cat is not None:
            cat.fit(X_train, y_train)
            fold_models["cat"] = cat
            m = eval_model(cat, X_test, y_test)
            print(f"  CatBoost -> Acc: {m['accuracy']:.4f}  F1: {m['f1']:.4f}  AUC: {m['auc']:.4f}")
            fold_records.append({"model": "CatBoost", "year": test_year, **m})
        else:
            fold_models["cat"] = None

                                                       
        lgb = build_lgb()
        if lgb is not None:
            lgb.fit(X_train, y_train)
            fold_models["lgb"] = lgb
            m = eval_model(lgb, X_test, y_test)
            print(f"  LightGBM -> Acc: {m['accuracy']:.4f}  F1: {m['f1']:.4f}  AUC: {m['auc']:.4f}")
            fold_records.append({"model": "LightGBM", "year": test_year, **m})
        else:
            fold_models["lgb"] = None

                                                       
        ens_prob = ensemble_proba(fold_models, ENS_WEIGHTS, X_test)
        ens_pred = (ens_prob >= 0.5).astype(int)
        m = {
            "accuracy":  accuracy_score(y_test, ens_pred),
            "precision": precision_score(y_test, ens_pred, zero_division=0),
            "recall":    recall_score(y_test, ens_pred, zero_division=0),
            "f1":        f1_score(y_test, ens_pred, zero_division=0),
            "auc":       roc_auc_score(y_test, ens_prob),
        }
        print(f"  Ensemble -> Acc: {m['accuracy']:.4f}  F1: {m['f1']:.4f}  AUC: {m['auc']:.4f}")
        fold_records.append({"model": "Ensemble", "year": test_year, **m})

                                                    
    results_df = pd.DataFrame(fold_records)
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS (mean ± std across all folds)")
    print("=" * 60)
    agg = results_df.groupby("model")[["accuracy", "precision", "recall", "f1", "auc"]].agg(["mean", "std"])
    print(agg.to_string())

                                                    
    print("\nTraining FINAL models on all data up to 2024, saving to models/...")
    final_train_years = [y for y in all_years if y < 2025]
    final_test_year   = max(y for y in all_years if y >= 2019)

    final_result = prepare_fold_data(df, final_train_years, final_test_year)
    if final_result is not None:
        X_tr, y_tr, X_te, y_te, _ = final_result

        final_xgb = build_xgb()
        final_xgb.fit(X_tr, y_tr)
        joblib.dump(final_xgb, os.path.join(MODELS_DIR, "xgboost_wfv.joblib"))
        print(f"  Saved xgboost_wfv.joblib  (test year {final_test_year})")

        if HAS_CATBOOST:
            final_cat = build_cat()
            final_cat.fit(X_tr, y_tr)
            joblib.dump(final_cat, os.path.join(MODELS_DIR, "catboost_model.joblib"))
            print(f"  Saved catboost_model.joblib")

        if HAS_LIGHTGBM:
            final_lgb = build_lgb()
            final_lgb.fit(X_tr, y_tr)
            joblib.dump(final_lgb, os.path.join(MODELS_DIR, "lightgbm_model.joblib"))
            print(f"  Saved lightgbm_model.joblib")

                                                    
    models_in_results = results_df["model"].unique()
    colors = {"XGBoost": "#e10600", "CatBoost": "#00b2ff", "LightGBM": "#2ecc71", "Ensemble": "#f39c12"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0e0f12")
    for ax in axes:
        ax.set_facecolor("#15161e")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#e0e6ed")
        for spine in ax.spines.values():
            spine.set_edgecolor("#222530")

    for model_name in models_in_results:
        sub = results_df[results_df["model"] == model_name].sort_values("year")
        c = colors.get(model_name, "#ffffff")
        lw = 2.5 if model_name == "Ensemble" else 1.5
        ls = "--" if model_name == "Ensemble" else "-"
        axes[0].plot(sub["year"], sub["f1"],       color=c, lw=lw, ls=ls, marker="o", label=model_name)
        axes[1].plot(sub["year"], sub["auc"],      color=c, lw=lw, ls=ls, marker="o", label=model_name)

    axes[0].set_title("F1 Score — Walk-Forward Folds",  fontsize=13, fontweight="bold")
    axes[1].set_title("ROC AUC — Walk-Forward Folds",   fontsize=13, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Test Year")
        ax.legend(facecolor="#1a1c24", edgecolor="#222530", labelcolor="#e0e6ed")
        ax.grid(alpha=0.15, color="#8b949e")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "16_walk_forward_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved chart: {out_path}")

                                                    
    mean_metrics = results_df.groupby("model")[["accuracy", "f1", "auc"]].mean().reset_index()
    x = np.arange(len(mean_metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0e0f12")
    ax.set_facecolor("#15161e")

    for i, metric in enumerate(["accuracy", "f1", "auc"]):
        bars = ax.bar(x + i * width, mean_metrics[metric], width=width,
                      label=metric.upper(), color=["#e10600", "#00b2ff", "#2ecc71"][i],
                      alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8, color="#e0e6ed")

    ax.set_xticks(x + width)
    ax.set_xticklabels(mean_metrics["model"], color="#e0e6ed")
    ax.tick_params(colors="#8b949e")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Mean Metrics Across Walk-Forward Folds", fontsize=14, color="#e0e6ed", fontweight="bold")
    ax.set_ylabel("Score", color="#8b949e")
    ax.legend(facecolor="#1a1c24", edgecolor="#222530", labelcolor="#e0e6ed")
    ax.grid(axis="y", alpha=0.15, color="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#222530")

    plt.tight_layout()
    out_path2 = os.path.join(FIGURES_DIR, "17_mean_metrics_comparison.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved chart: {out_path2}")

    print("\nWalk-Forward Evaluation complete!")


if __name__ == "__main__":
    main()
