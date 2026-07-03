import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV

                  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model_trainer import prepare_ml_data, evaluate_predictions

def train_and_evaluate():
    print("Loading data for model training...")
    df = pd.read_csv("data/processed/model_ready.csv")
    
                         
    print("Preparing train/test split (split year = 2019)...")
    X_train, y_train, X_test, y_test, X_train_s, X_test_s, preprocessor = prepare_ml_data(df, split_year=2019)
    
                       
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(preprocessor, f"{models_dir}/preprocessor.joblib")
    print(f"Saved preprocessor metadata to {models_dir}/preprocessor.joblib")
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Training Top10 rate: {y_train.mean():.2%}, Test Top10 rate: {y_test.mean():.2%}")
    
    results = {}
    
                                                          
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_preds = lr.predict(X_test_s)
    lr_probs = lr.predict_proba(X_test_s)[:, 1]
    results["Logistic Regression"] = {
        "model": lr,
        "metrics": evaluate_predictions(y_test, lr_preds, lr_probs),
        "probs": lr_probs
    }
    
                                                          
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=6, min_samples_split=20, random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_probs = dt.predict_proba(X_test)[:, 1]
    results["Decision Tree"] = {
        "model": dt,
        "metrics": evaluate_predictions(y_test, dt_preds, dt_probs),
        "probs": dt_probs
    }
    
                                                          
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    results["Random Forest"] = {
        "model": rf,
        "metrics": evaluate_predictions(y_test, rf_preds, rf_probs),
        "probs": rf_probs
    }
    
                                                          
    print("Training Baseline XGBoost...")
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    results["XGBoost (Baseline)"] = {
        "model": xgb,
        "metrics": evaluate_predictions(y_test, xgb_preds, xgb_probs),
        "probs": xgb_probs
    }
    
                                                          
    print("Tuning XGBoost Hyperparameters...")
    param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9]
    }
    xgb_tuning = XGBClassifier(random_state=42, n_jobs=-1)
                                                                                        
    rs = RandomizedSearchCV(xgb_tuning, param_distributions=param_dist, n_iter=8, scoring="roc_auc", cv=3, random_state=42, n_jobs=-1)
    rs.fit(X_train, y_train)
    print(f"Best XGBoost Params: {rs.best_params_}")
    
    xgb_tuned = rs.best_estimator_
    xgb_tuned_preds = xgb_tuned.predict(X_test)
    xgb_tuned_probs = xgb_tuned.predict_proba(X_test)[:, 1]
    results["XGBoost (Tuned)"] = {
        "model": xgb_tuned,
        "metrics": evaluate_predictions(y_test, xgb_tuned_preds, xgb_tuned_probs),
        "probs": xgb_tuned_probs
    }
    
                 
    for name, res in results.items():
        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        filepath = f"{models_dir}/{filename}.joblib"
        joblib.dump(res["model"], filepath)
        print(f"Saved {name} model to {filepath}")
        
                            
    metrics_summary = []
    for name, res in results.items():
        m = res["metrics"]
        metrics_summary.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 Score": m["f1"],
            "ROC AUC": m["auc"]
        })
        
    summary_df = pd.DataFrame(metrics_summary)
    print("\nModel Comparison Table:")
    print(summary_df.to_string(index=False))
    
                        
    summary_df.to_csv("reports/model_comparison.csv", index=False)
    
                                                          
    fig_dir = "reports/figures"
    
                    
    plt.figure(figsize=(10, 8))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["probs"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison (Test Set)', fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/11_roc_comparison.png", dpi=150, facecolor='#121212')
    plt.close()
    
                            
    plt.figure(figsize=(12, 6))
    melted_df = pd.melt(summary_df, id_vars="Model", var_name="Metric", value_name="Value")
    sns.barplot(data=melted_df, x="Metric", y="Value", hue="Model", palette="coolwarm")
    plt.title("Performance Metrics Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/12_metrics_comparison.png", dpi=150, facecolor='#121212')
    plt.close()
    
    print("\nVisualizations saved successfully to reports/figures/")

if __name__ == "__main__":
    train_and_evaluate()
