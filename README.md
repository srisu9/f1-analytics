# 🏎️ F1 Analytics AI Platform — Version 4

> **XGBoost walk-forward validated race prediction engine | Ergast + FastF1 | SHAP-explained**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-FF6600?logo=xgboost)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Overview

An end-to-end machine learning platform that predicts **Formula 1 Top-10 race finishes** using a production-grade XGBoost classifier trained on 30+ years of historical race data (1994–2024). The platform combines Ergast CSV data with FastF1 telemetry, applies Bayesian-smoothed circuit history, and provides SHAP-explainable predictions through a polished multi-tab Streamlit dashboard.

**This is not a toy project.** The pipeline addresses real ML engineering challenges:
- Strict **temporal data leakage prevention** via walk-forward cross-validation
- **Bayesian smoothing** of sparse circuit statistics
- **Training/inference parity** enforced via stored imputation medians and target encoders
- Automated **leakage detection** wired into the training pipeline

---

## Architecture

```
F1 Analytics V4
├── data/
│   ├── raw/                  ← Ergast CSV files (not tracked in Git)
│   ├── processed/
│   │   └── model_ready.csv   ← Engineered feature dataset (all phases)
│   └── fastf1/               ← FastF1 cache (not tracked in Git)
│
├── src/
│   ├── config.py             ← All paths, constants, and version (single source of truth)
│   ├── data_loader.py        ← Ergast CSV loaders and merge logic
│   ├── feature_engineer.py   ← Core feature computation (Phase 0 baseline)
│   ├── leakage_checker.py    ← Automated leakage auditing
│   ├── shap_reporter.py      ← SHAP beeswarm and importance plots
│   └── features/
│       ├── phase2_circuit_history.py  ← Bayesian-smoothed circuit history
│       ├── phase3_weather_safety.py   ← Historical safety car rate
│       └── phase5_championship.py    ← Championship standings & rolling form
│
├── notebooks/
│   └── v4_training_pipeline.py  ← Central training pipeline (all phases)
│
├── models/
│   ├── v4_xgb_final.joblib      ← Champion model (Phase 2 features)
│   ├── v4_phase2_xgb.joblib     ← Phase 2 model artifact
│   ├── v4_phase5_xgb.joblib     ← Phase 5 model artifact
│   └── preprocessor.joblib      ← Encoders + medians (training/inference parity)
│
├── app/
│   └── streamlit_app.py         ← 5-tab interactive dashboard
│
├── tests/
│   ├── test_feature_engineer.py ← Feature leakage and correctness tests
│   └── test_inference_pipeline.py ← Inference pipeline parity tests
│
└── reports/
    ├── v3_baseline_metrics.json
    ├── v4_phase2_metrics.json
    ├── v4_phase5_metrics.json
    └── v4_phase3_metrics.json
```

---

## Data Flow

```
Ergast CSV (raw)
      │
      ▼
merge_datasets()          ← joins results, races, drivers, constructors, circuits
      │
      ▼
engineer_features()       ← rolling stats, cumulative career rates, home_race, grid_diff
      │
      ▼
add_phase2_features()     ← Bayesian-smoothed circuit history, circuit overtaking index
      │
      ▼
walk_forward_evaluate()   ← train on years < T, test on year T (2019–2024)
      │
      ├──── target_encode()  ← driverRef, constructor_name, circuitRef (fit on train only)
      │
      ├──── Impute NaN with column medians (stored in preprocessor.joblib)
      │
      └──── XGBClassifier.fit()
```

---

## Feature Engineering

All features are strictly **pre-race** (no leakage):

| Feature | Description | Source |
|---|---|---|
| `grid` | Starting grid position | Ergast results |
| `driver_age` | Age on race day | Ergast drivers |
| `driver_experience` | Career starts before this race | Ergast (cumcount) |
| `driver_top10_rate` | Rolling career Top-10 rate (Bayesian prior = 0.50) | Ergast (cumsum/shift) |
| `constructor_top10_rate` | Rolling team Top-10 rate | Ergast (cumsum/shift) |
| `rolling_avg_finish_3` | 3-race rolling average finish | Ergast (shift) |
| `rolling_avg_finish_5` | 5-race rolling average finish | Ergast (shift) |
| `prev_race_finish` | Previous race finishing position | Ergast (shift) |
| `home_race` | 1 if driver racing in home country | Ergast + nationality map |
| `grid_qualifying_diff` | Grid vs qualifying position delta | Ergast |
| `constructor_season_points` | Cumulative team points before this race | Ergast (cumsum/shift) |
| `lat`, `lng`, `alt` | Circuit geolocation | Ergast circuits |
| `smoothed_circuit_avg_finish` | Bayesian-smoothed driver avg finish at this circuit | Ergast (Phase 2) |
| `circuit_grid_finish_corr` | Rolling 3-year grid-to-finish correlation | Ergast (Phase 2) |
| `driverRef_encoded` | Smoothed target encoding of driver | Ergast (fit on train) |
| `constructor_name_encoded` | Smoothed target encoding of constructor | Ergast (fit on train) |
| `circuitRef_encoded` | Smoothed target encoding of circuit | Ergast (fit on train) |

### Bayesian Smoothing

Circuit history uses Bayesian smoothing to handle sparse data (e.g., a driver's first race at Monaco):

```
smoothed = (n_visits × circuit_avg + K × career_avg) / (n_visits + K)
```

Where `K=4` is the smoothing factor. The `career_avg` is a **global career average** (across all circuits), not accidentally circuit-specific.

---

## Walk-Forward Validation

The model is trained and evaluated using strict temporal cross-validation to prevent look-ahead bias:

- **Training window**: All races from 1994 up to year `T-1`
- **Test window**: All races in year `T`
- **Test years**: 2019, 2020, 2021, 2022, 2023, 2024

Metrics are averaged across all 6 test years. The final production model is retrained on the full dataset (1994–2024).

---

## Results

Walk-forward averaged metrics (2019–2024):

| Model | Accuracy | F1 | ROC-AUC | Notes |
|---|---|---|---|---|
| V3 Baseline (14 features) | 0.7737 | 0.7788 | 0.8372 | Baseline |
| V4 Phase 2 (+Circuit History) | 0.7731 | 0.7775 | 0.8367 | **Champion model** |
| V4 Phase 5 (+Championship Form) | 0.7746 | 0.7788 | 0.8394 | Experimental |
| V4 Phase 3 (+Weather/SC Rate) | 0.7777 | 0.7832 | 0.8390 | Requires FastF1 |

> **Why is Phase 2 the champion despite slightly lower raw metrics?**
> Phase 2 uses only Ergast CSV data — no external API dependencies, no FastF1 cache required.
> It is the most reproducible, robust, and deployable model. Phase 5 and Phase 3 add marginal
> gains at the cost of complexity and data pipeline requirements.

---

## Model Limitations

This model is designed for **historical analysis and entertainment**. Key limitations:

- Predicts **Top-10 probability**, not exact finishing position
- Does not model: mechanical failures, safety car timing, pit strategy errors, crashes
- Weather simulation is a heuristic overlay — not a physics simulation
- FastF1 data (qualifying pace, weather) improves predictions but adds API dependency
- Season Projector over-rewards consistent midfield finishers (probability ≠ race pace)

---

## FastF1 Integration

FastF1 is used for practice and qualifying session data. To populate the cache:

```python
import fastf1
cache_dir = "data/fastf1"
fastf1.Cache.enable_cache(cache_dir)

session = fastf1.get_session(2024, "Bahrain", "Q")
session.load()
```

The Phase 3 pipeline uses historical weather and safety car rate from FastF1 telemetry.

---

## SHAP Explainability

Every prediction is explainable using SHAP (SHapley Additive exPlanations):

- **Beeswarm plots** show global feature importance across all races
- **Waterfall plots** show individual driver prediction breakdowns in the UI
- The most predictive features are: `grid`, `driverRef_encoded`, `rolling_avg_finish_3`, `constructor_name_encoded`

---

## Installation

```bash
# Clone the repo
git clone https://github.com/srisu9/f1-analytics.git
cd f1-analytics

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Place Ergast CSV files in data/raw/
# (Download from: http://ergast.com/mrd/)
```

---

## Running the Training Pipeline

```bash
# Run all phases (baseline + Phase 2 + Phase 5)
python notebooks/v4_training_pipeline.py --phase 0 2 5

# Run just the champion Phase 2 model
python notebooks/v4_training_pipeline.py --phase 0 2

# This saves:
#   models/v4_xgb_final.joblib     ← Champion model
#   models/preprocessor.joblib     ← Encoders + imputation medians
#   data/processed/model_ready.csv ← Engineered feature dataset
#   reports/*.json                 ← Walk-forward metrics per phase
```

---

## Running the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard provides 5 tabs:
1. **Grid Predictor** — Predict Top-10 probabilities for any race (with weather/SC simulation)
2. **Head-to-Head** — Compare two drivers at a selected circuit
3. **Historical Replay** — Run the model on a past race and compare vs. actual results
4. **Season Projector** — Simulate an entire season's championship standings
5. **Model Evolution** — Phase-by-phase metric comparison and SHAP importance plots

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover:
- Feature leakage removal (positionOrder, points, time)
- Target column creation (Top10)
- Bayesian prior for new drivers (GLOBAL_TOP10_PRIOR = 0.50)
- Rolling average with shift (no look-ahead)
- Inference pipeline: target encoding, missing column imputation, feature order

---

## Future Work

- **FastF1 2025 live data**: Connect to OpenF1 API for real-time practice/qualifying data
- **Incident modelling**: Integrate retirement probability as a separate target
- **Ordinal prediction**: Predict finishing position rather than binary Top-10
- **Constructor confidence intervals**: Model uncertainty via ensemble or calibration
- **Live race dashboard**: Real-time probability updates using lap timing data

---

## Data Sources

- **Ergast Developer API** — Historical race results, drivers, constructors, circuits (1950–2024)
- **FastF1** — Telemetry, session timing, weather data
- **OpenF1 API** — Real-time 2024+ race data (experimental)

---

## License

MIT License — See [LICENSE](LICENSE) for details.
