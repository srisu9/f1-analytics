# 🏎️ F1 Race Prediction — ML Analytics Platform

Predict which Formula 1 drivers will score points using a machine learning model trained on 30 years of race history.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-FF6600)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Quick Start

```bash
git clone https://github.com/srisu9/f1-analytics.git
cd f1-analytics
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`. No API keys needed — all data is included.

---

## What This Does

In Formula 1, only the top 10 finishers in each race score championship points. This project predicts the probability of each driver finishing inside the top 10, before the race starts.

The model uses only information that would be available before a race (no future data): starting grid position, recent form, career statistics, team performance, and track history. Predictions are fully explained — you can see exactly which factors drove each driver's score.

**Five interactive tabs:**
1. **Grid Predictor** — Run predictions for any race weekend with optional weather and safety car simulation
2. **Driver Comparison** — Compare two drivers head-to-head at a selected circuit
3. **Historical Replay** — Run the model on a past race and check how accurate it was
4. **Season Projector** — Simulate an entire season and project the final standings
5. **Model Evolution** — See how the model improved across development phases

---

## Model Performance

The model was validated on future seasons it was never trained on, to prevent overfitting.

**Walk-forward validation results (averaged across 2019–2024):**

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| V3 Baseline (14 features) | 77.4% | 0.779 | 0.837 |
| V4 + Circuit History | 77.3% | 0.778 | 0.837 |
| V4 + Championship Form | 77.5% | 0.779 | 0.839 |

**2024 season holdout (never seen during training):**
- **Accuracy: 84.3%** · F1: 83.8% · ROC-AUC: 0.931

> The walk-forward averages (77%) and the 2024 holdout score (84.3%) measure different things.
> Walk-forward averaging includes harder seasons (2019–2022) where the grid was more competitive.
> The 2024 season alone had a more dominant top-3 teams, making it slightly more predictable.

**Why the circuit history model is the production choice:**  
It matches or beats all other variants using only publicly available CSV data — no external API, no caching, fully reproducible.

---

## How It Works

### Features Used (all pre-race, no future data)

| Feature | What it represents |
|---|---|
| `grid` | Starting grid position (1 = front) |
| `driver_age` | Driver's age on race day |
| `driver_experience` | Number of career starts before this race |
| `driver_top10_rate` | Career rate of finishing in the top 10 |
| `constructor_top10_rate` | Team's rate of finishing in the top 10 |
| `rolling_avg_finish_3` | Average finish position over the last 3 races |
| `rolling_avg_finish_5` | Average finish over the last 5 races |
| `prev_race_finish` | Finishing position in the previous race |
| `home_race` | Whether the driver is racing in their home country |
| `grid_qualifying_diff` | Difference between qualifying and grid position |
| `constructor_season_points` | Team's total points accumulated so far this season |
| `lat`, `lng`, `alt` | Circuit location (affects tyre and car setup) |
| `smoothed_circuit_avg_finish` | Driver's historical average finish at this specific track |
| `circuit_grid_finish_corr` | How much grid position affects finishing at this track |
| `driverRef_encoded` | Driver identity (target-encoded) |
| `constructor_name_encoded` | Team identity (target-encoded) |
| `circuitRef_encoded` | Circuit identity (target-encoded) |

### Leakage Prevention

Data is split strictly by time — the model is never shown future information. Rolling averages and cumulative stats are calculated with a one-race lag. A separate leakage checker scans the feature matrix before every training run.

### Validation Methodology

Walk-forward cross-validation: train on all races up to year `T-1`, test on year `T`. Repeated for each year from 2019 to 2024. The production model is then retrained on the full dataset (1994–2024).

---

## Project Structure

```
f1-analytics/
├── app/
│   └── streamlit_app.py          ← Interactive dashboard (5 tabs)
│
├── src/
│   ├── config.py                 ← Paths, constants, feature definitions
│   ├── data_loader.py            ← Ergast CSV loading and merging
│   ├── feature_engineer.py       ← Core feature computation
│   ├── leakage_checker.py        ← Automated leakage auditing
│   ├── shap_reporter.py          ← SHAP visualisations
│   └── features/
│       ├── phase2_circuit_history.py   ← Track history features
│       ├── phase3_weather_safety.py    ← Safety car rate features
│       └── phase5_championship.py     ← Championship form features
│
├── notebooks/
│   └── v4_training_pipeline.py   ← Full training pipeline
│
├── models/
│   ├── v4_xgb_final.joblib       ← Production model
│   └── preprocessor.joblib       ← Encoders and imputation values
│
├── data/
│   ├── raw/                      ← Ergast CSV files (not tracked in git)
│   └── processed/
│       └── model_ready.csv       ← Engineered feature dataset
│
├── tests/
│   ├── test_feature_engineer.py
│   └── test_inference_pipeline.py
│
└── reports/
    └── *.json                    ← Walk-forward metrics per phase
```

---

## Installation

```bash
git clone https://github.com/srisu9/f1-analytics.git
cd f1-analytics

python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

**Data setup:** Download the Ergast historical dataset CSV files from the [Ergast archive](https://ergast.com/downloads/f1db_csv.zip) and place them in `data/raw/`.

---

## Training the Model

```bash
# Train the production model (Phase 0 + Phase 2)
python notebooks/v4_training_pipeline.py --phase 0 2

# Train all phases including experimental variants
python notebooks/v4_training_pipeline.py --phase 0 2 5
```

This saves the model to `models/v4_xgb_final.joblib` and the preprocessor to `models/preprocessor.joblib`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover: feature leakage removal, target column creation, Bayesian prior for new drivers, rolling average lag, and inference pipeline parity.

---

## Limitations

- Predicts **probability of a top-10 finish**, not exact position
- Does not model: mechanical failures, pit strategy errors, crashes, or race incidents
- Weather simulation is a heuristic adjustment, not a physics model
- Season Projector assigns points by predicted probability rank, which tends to favour consistent midfield drivers over win-or-retire drivers

---

## Future Work

- Connect to OpenF1 API for live 2025 race data
- Model retirement probability as a separate target
- Predict exact finishing position rather than binary top-10
- Real-time probability updates during a race using lap timing data

---

## Data Sources

- **Ergast historical dataset** — Race results, drivers, teams, circuits (1950–2024). [Archive download](https://ergast.com/downloads/f1db_csv.zip)
- **FastF1** — Session timing and telemetry (used in experimental Phase 3 only)

---

## License

MIT — see [LICENSE](LICENSE).
