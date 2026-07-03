# 🏎️ F1 Analytics AI Platform (Version 3)

An end-to-end machine learning platform that predicts Formula 1 race outcomes for entire 20-driver grids using XGBoost, CatBoost, LightGBM, and a Soft-Voting Ensemble.

## Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://f1-analytics.streamlit.app)

---

## Features

| Feature | Description |
|---|---|
| 🏁 **Full Grid Predictor** | Predicts and ranks all 20 drivers by probability of a Top 10 finish |
| ☀️ **Race Simulator** | Adjustable weather (Dry/Mixed/Wet) and safety car probability overlays |
| 🔍 **Live SHAP Explanations** | Per-driver live SHAP waterfall charts explaining each prediction |
| ⚔️ **Head-to-Head** | Compare any two drivers and calculate who is likely to finish ahead |
| ⏳ **Historical Replays** | Run the model on past races and compare predictions vs. actual results |
| 🏆 **Season Predictor** | Simulate an entire championship season and project final standings |

---

## Model Architecture

- **Default Model**: Tuned XGBoost (Walk-Forward Validated)
- **Ensemble**: Soft-Voting Blender (XGBoost × 0.4 + CatBoost × 0.4 + LightGBM × 0.2)
- **Validation Strategy**: Chronological Walk-Forward CV (2019–2024)

| Model | Accuracy | F1 Score | ROC AUC |
|---|---|---|---|
| CatBoost | 77.77% | 78.13% | 0.8384 |
| Ensemble | 77.74% | **78.21%** | 0.8372 |
| XGBoost | 77.24% | 77.86% | 0.8344 |
| LightGBM | 76.94% | 77.47% | 0.8308 |

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/f1-analytics-ai-platform.git
cd f1-analytics-ai-platform

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
f1-analytics-ai-platform/
├── app/
│   └── streamlit_app.py          # Main Streamlit dashboard (4 tabs)
├── data/
│   ├── raw/                      # Original Ergast F1 CSVs
│   └── processed/
│       └── model_ready.csv       # Engineered features dataset
├── models/
│   ├── xgboost_wfv.joblib        # Walk-forward trained XGBoost
│   ├── catboost_model.joblib     # CatBoost model
│   ├── lightgbm_model.joblib     # LightGBM model
│   ├── random_forest.joblib      # Random Forest model
│   └── preprocessor.joblib       # Target encoders & scaler
├── notebooks/
│   ├── 02_data_cleaning.py
│   ├── 03_eda.py
│   ├── 04_feature_engineering.py
│   ├── 05_model_training.py
│   ├── 06_explainability.py
│   ├── 07_error_analysis.py
│   └── 08_walk_forward_eval.py
├── src/
│   ├── model_trainer.py          # Data prep & feature engineering
│   └── ensemble.py               # Soft-voting ensemble blender
├── reports/
│   └── figures/                  # 20 EDA, model, and SHAP charts
├── requirements.txt
└── README.md
```

---

## Data Source
Raw F1 datasets are sourced from the [Ergast Motor Racing API](https://ergast.com/mrd/) via the [F1 Kaggle Dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020). The dataset covers seasons from 1950 to 2024.
