import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

# Add src/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model_trainer import prepare_ml_data

def run_explainability():
    print("Loading model and data for SHAP analysis...")
    df = pd.read_csv("data/processed/model_ready.csv")
    
    # Pre-process & Split
    X_train, y_train, X_test, y_test, _, _, preprocessor = prepare_ml_data(df, split_year=2019)
    
    # Load the best model (using XGBoost Tuned as primary, fallback to Random Forest)
    model_path = "models/xgboost_tuned.joblib"
    if not os.path.exists(model_path):
        model_path = "models/random_forest.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found in models/ directory. Run model training first.")
        
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path} for explanation.")
    
    # Create figures directory
    fig_dir = "reports/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # Calculate SHAP values
    # ----------------------------------------------------
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Sample a subset of test data for explanation to run fast
    sample_size = min(500, len(X_test))
    print(f"Calculating SHAP values for a sample of {sample_size} test rows...")
    X_sample = X_test.iloc[:sample_size]
    
    shap_values = explainer(X_sample)
    
    # ----------------------------------------------------
    # 1. SHAP Summary Plot
    # ----------------------------------------------------
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    # SHAP plot uses its own figure control, but we pass show=False to save it
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Global Feature Importance & Impact (Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/13_shap_summary.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # ----------------------------------------------------
    # 2. SHAP Waterfall Plot (Individual Prediction)
    # ----------------------------------------------------
    print("Generating SHAP Waterfall Plot for a single prediction...")
    plt.figure(figsize=(10, 6))
    # Let's take the first row of sample (an interesting prediction)
    # shap.plots.waterfall requires a Explanation object
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("SHAP Individual Prediction Explanation (First Test Sample)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/14_shap_waterfall.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # ----------------------------------------------------
    # 3. SHAP Dependence Plot (Grid Position)
    # ----------------------------------------------------
    print("Generating SHAP Dependence Plot for 'grid'...")
    plt.figure(figsize=(10, 6))
    # find index of grid feature
    grid_idx = X_sample.columns.get_loc("grid")
    # Plot dependency
    shap.dependence_plot("grid", shap_values.values, X_sample, show=False)
    plt.title("SHAP Dependence Plot: Starting Grid Position", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/15_shap_dependence.png", dpi=150, facecolor='#121212')
    plt.close()
    
    print("SHAP plots generated and saved successfully to reports/figures/")

if __name__ == "__main__":
    run_explainability()
