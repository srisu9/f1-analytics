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
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.model_trainer import prepare_ml_data

                                                
DATA_PATH   = "data/processed/model_ready.csv"
MODELS_DIR  = "models"
FIGURES_DIR = "reports/figures"
SPLIT_YEAR  = 2019
FP_THRESH   = 0.75                                                 
FN_THRESH   = 0.25                                                        
                                                

os.makedirs(FIGURES_DIR, exist_ok=True)


def set_dark_style(fig, axes):
    """Apply consistent dark theme to a matplotlib figure."""
    fig.patch.set_facecolor("#0e0f12")
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor("#15161e")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#e0e6ed")
        for spine in ax.spines.values():
            spine.set_edgecolor("#222530")


def main():
    print("=" * 60)
    print("F1 Analytics — Error Analysis & Biggest Upsets")
    print("=" * 60)

                                                                    
    df_raw = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df_raw):,} rows.")

    X_train, y_train, X_test, y_test, X_train_sc, X_test_sc, preprocessor = prepare_ml_data(
        df_raw, split_year=SPLIT_YEAR
    )

                                                                    
    for path in [
        os.path.join(MODELS_DIR, "xgboost_wfv.joblib"),
        os.path.join(MODELS_DIR, "xgboost_tuned.joblib"),
        os.path.join(MODELS_DIR, "random_forest.joblib"),
    ]:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"Loaded model: {path}")
            break
    else:
        raise FileNotFoundError("No trained model found. Run training pipeline first.")

                                                                    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

                                                               
    test_mask = df_raw["year"] >= SPLIT_YEAR
    df_test   = df_raw[test_mask].copy().reset_index(drop=True)
    df_test   = df_test.iloc[:len(y_prob)].copy()                          

    df_test["pred_prob"]  = y_prob
    df_test["pred_label"] = y_pred
    df_test["actual"]     = y_test.values[:len(y_prob)]
    df_test["correct"]    = (df_test["pred_label"] == df_test["actual"]).astype(int)
    df_test["error_mag"]  = (df_test["pred_prob"] - df_test["actual"]).abs()

                                                                    
    fp_mask = (df_test["pred_prob"] >= FP_THRESH) & (df_test["actual"] == 0)
    df_fp   = df_test[fp_mask].sort_values("pred_prob", ascending=False)

    print(f"\n{'='*60}")
    print(f"FALSE POSITIVES (High-confidence Top10 predicted, but finished outside)")
    print(f"Threshold: P(Top10) >= {FP_THRESH}")
    print(f"Total upsets found: {len(df_fp)}")
    print(f"{'='*60}")
    cols_show = ["year", "race_name", "driverRef", "constructor_name", "grid", "pred_prob"]
    print(df_fp[cols_show].head(20).to_string(index=False))

                                                                       
    fn_mask = (df_test["pred_prob"] <= FN_THRESH) & (df_test["actual"] == 1)
    df_fn   = df_test[fn_mask].sort_values("pred_prob", ascending=True)

    print(f"\n{'='*60}")
    print(f"FALSE NEGATIVES (Low-confidence predicted, but driver DID finish Top10)")
    print(f"Threshold: P(Top10) <= {FN_THRESH}")
    print(f"Total comebacks found: {len(df_fn)}")
    print(f"{'='*60}")
    print(df_fn[cols_show].head(20).to_string(index=False))

                                                                    
    brier = brier_score_loss(df_test["actual"], df_test["pred_prob"])
    print(f"\nBrier Score: {brier:.4f}  (lower = better calibrated; perfect = 0.0)")

                                                                     
    frac_pos, mean_pred = calibration_curve(df_test["actual"], df_test["pred_prob"], n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    set_dark_style(fig, ax)
    ax.plot([0, 1], [0, 1], "k--", color="#8b949e", lw=1, label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="#ff1801", lw=2, label=f"Model (Brier={brier:.3f})")
    ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.15, color="#ff1801")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Top10 Rate)")
    ax.set_title("Probability Calibration Curve", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1c24", edgecolor="#222530", labelcolor="#e0e6ed")
    ax.grid(alpha=0.15, color="#8b949e")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "18_error_calibration.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved: {out}")

                                                                     
    by_year = df_test.groupby("year").apply(lambda g: pd.Series({
        "False Positives":  ((g["pred_prob"] >= FP_THRESH) & (g["actual"] == 0)).sum(),
        "False Negatives":  ((g["pred_prob"] <= FN_THRESH) & (g["actual"] == 1)).sum(),
        "Correct (≥75%)":   ((g["pred_prob"] >= FP_THRESH) & (g["actual"] == 1)).sum(),
    })).reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    set_dark_style(fig, ax)
    x = np.arange(len(by_year))
    w = 0.28
    ax.bar(x - w, by_year["False Positives"], width=w, label="False Positives",  color="#ff1801", alpha=0.85)
    ax.bar(x,     by_year["False Negatives"], width=w, label="False Negatives",  color="#00b2ff", alpha=0.85)
    ax.bar(x + w, by_year["Correct (≥75%)"], width=w, label="Correct (≥75%)",   color="#2ecc71", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(by_year["year"].astype(int), color="#e0e6ed")
    ax.set_title("High-Confidence Predictions by Year", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.legend(facecolor="#1a1c24", edgecolor="#222530", labelcolor="#e0e6ed")
    ax.grid(axis="y", alpha=0.15, color="#8b949e")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "19_error_by_year.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")

                                                                    
    top_upsets = df_fp.head(20).copy()
    top_upsets["label"] = (
        top_upsets["driverRef"] + "\n" +
        top_upsets["race_name"].str[:15] + " " +
        top_upsets["year"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    set_dark_style(fig, ax)
    bars = ax.barh(top_upsets["label"], top_upsets["pred_prob"],
                   color="#ff1801", alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted P(Top10) — Model was confident but WRONG")
    ax.set_title("Top 20 Biggest Upsets (False Positives)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(0.5, color="#8b949e", lw=1, ls="--")
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.0%}", va="center", fontsize=8, color="#e0e6ed")
    ax.grid(axis="x", alpha=0.15, color="#8b949e")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "20_top_upsets.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")

    print("\nError Analysis complete!")
    print(f"\nSummary:")
    print(f"  Total test samples    : {len(df_test):,}")
    print(f"  Overall accuracy      : {df_test['correct'].mean():.2%}")
    print(f"  High-conf. FP upsets  : {len(df_fp):,}  ({len(df_fp)/len(df_test):.1%} of test set)")
    print(f"  High-conf. FN comebacks: {len(df_fn):,}  ({len(df_fn)/len(df_test):.1%} of test set)")
    print(f"  Brier score           : {brier:.4f}")


if __name__ == "__main__":
    main()
