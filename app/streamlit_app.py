import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import WEATHER_SIM, PROJECT_VERSION

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Analytics AI Platform v4",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #080810; color: #e0e6ed; }

h1, h2, h3, h4, h5 { color: #ff1801 !important; font-weight: 800; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e0f18 0%, #12131f 100%);
    border-right: 1px solid #1e2035;
}

.version-tag {
    background: linear-gradient(90deg, #ff1801, #cc0000);
    color: white; font-size: 12px; font-weight: 800;
    padding: 4px 12px; border-radius: 20px;
    display: inline-block; margin-left: 12px; letter-spacing: 1px;
}

.disclaimer-box {
    background: rgba(255,24,1,0.08); border-left: 4px solid #ff1801;
    padding: 10px 16px; border-radius: 8px; margin: 12px 0;
    font-size: 13px; color: #fca5a5;
}

div[data-testid="stTabs"] button { color: #6e7a94 !important; font-weight: 600; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #ff1801 !important; border-bottom-color: #ff1801 !important; }

/* Override Streamlit metric card styling */
[data-testid="stMetric"] {
    background: #13141f;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #1e2035;
}
[data-testid="stMetricValue"] { color: #ff1801 !important; font-size: 2rem !important; font-weight: 900 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
# These are the V4 Phase 2 features that must exist in model_ready.csv
V4_CIRCUIT_FEATURES = ["smoothed_circuit_avg_finish", "circuit_grid_finish_corr"]

# Default values used when constructing synthetic rows (H2H tab)
V4_CIRCUIT_DEFAULTS = {
    "smoothed_circuit_avg_finish": 10.0,  # mid-field default
    "circuit_grid_finish_corr":    0.50,  # neutral overtaking
}


# ─────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/model_ready.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


@st.cache_resource
def load_preprocessor():
    return joblib.load("models/preprocessor.joblib")


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        st.stop()
    return joblib.load(path)


@st.cache_resource
def build_shap_explainer(_model):
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return None


@st.cache_data
def load_phase_metrics():
    phases = {
        "V3 Baseline":              "reports/v3_baseline_metrics.json",
        "Phase 2 (Circuit Hist.)":  "reports/v4_phase2_metrics.json",
        "Phase 5 (Champ. Form)":    "reports/v4_phase5_metrics.json",
        "Phase 3 (Weather+SC)":     "reports/v4_phase3_metrics.json",
    }
    result = {}
    for label, path in phases.items():
        if os.path.exists(path):
            with open(path) as f:
                result[label] = json.load(f)
    return result


# ─────────────────────────────────────────────────────────
# Feature processing
# ─────────────────────────────────────────────────────────
def process_features(df_in: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    """
    Applies target encoding and returns a DataFrame with exactly the
    columns the V4 model expects, in the correct order.
    """
    enc = preprocessor["encoders"]
    feature_cols = preprocessor["features"]
    medians = preprocessor.get("imputation_medians", {})

    df_p = df_in.copy().reset_index(drop=True)

    # Target-encode categoricals
    for cat in ["driverRef", "constructor_name", "circuitRef"]:
        if cat in df_p.columns:
            smooth_map  = enc[cat]["map"]
            global_mean = enc[cat]["global_mean"]
            df_p[cat + "_encoded"] = df_p[cat].map(smooth_map).fillna(global_mean)

    # Keep only expected columns, in the right order; fill any missing NaN
    missing = [c for c in feature_cols if c not in df_p.columns]
    if missing:
        st.warning(f"Warning: Expected features missing from input data: {missing}. Imputing with training medians.")
        for c in missing:
            df_p[c] = medians.get(c, 0.0)

    # For existing columns that might contain NaN, impute with training median
    for col in df_p.columns:
        if col in medians:
            df_p[col] = df_p[col].fillna(medians[col])

    return df_p[feature_cols].fillna(0.0)


# ─────────────────────────────────────────────────────────
# SHAP waterfall
# ─────────────────────────────────────────────────────────
def plot_shap_waterfall(explainer, row_array: np.ndarray, feature_names: list, driver_name: str):
    """row_array must be shape (1, n_features)."""
    if explainer is None:
        st.info("SHAP not available.")
        return
    try:
        sv = explainer(row_array)
        if hasattr(sv, "values") and sv.values.ndim == 3:
            sv = shap.Explanation(
                values=sv.values[0, :, 1],
                base_values=sv.base_values[0, 1] if sv.base_values.ndim > 1 else sv.base_values[0],
                data=sv.data[0], feature_names=feature_names
            )
        elif hasattr(sv, "values") and sv.values.ndim == 2:
            sv = shap.Explanation(
                values=sv.values[0],
                base_values=sv.base_values[0] if hasattr(sv.base_values, "__len__") else sv.base_values,
                data=sv.data[0], feature_names=feature_names
            )
        fig, _ = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#13141f")
        shap.plots.waterfall(sv, max_display=8, show=False)
        for ax in fig.get_axes():
            ax.set_facecolor("#13141f")
            ax.tick_params(colors="#e0e6ed", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2035")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.caption(f"SHAP unavailable: {e}")


# ─────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────
def apply_weather_override(df: pd.DataFrame, weather: str) -> pd.DataFrame:
    df_s = df.copy()
    
    if weather == "Dry":
        return df_s
        
    modifiers = WEATHER_SIM.get(weather, {})
    
    for feature, modifier in modifiers.items():
        if feature in df_s.columns:
            if isinstance(modifier, float) and modifier < 1.0:
                # Multiplicative penalty (e.g. 0.90 for reliability drop)
                df_s[feature] = df_s[feature] * modifier
            else:
                # Additive penalty (e.g. +3.0 to average finish for pace drop)
                df_s[feature] = df_s[feature] + modifier
                
    # Grid always converges slightly in wet (driver skill > car pace)
    if weather == "Mixed":
        df_s["grid"] = 10.5 + 0.7 * (df_s["grid"] - 10.5)
    elif weather == "Wet":
        df_s["grid"] = 10.5 + 0.45 * (df_s["grid"] - 10.5)
        
    return df_s


def apply_sc_noise(probs: np.ndarray, sc_level: str) -> np.ndarray:
    np.random.seed(42)
    n = len(probs)
    if sc_level == "Medium":
        probs = np.clip(probs + np.random.normal(0, 0.04, n), 0.01, 0.99)
    elif sc_level == "High":
        probs = np.clip(probs + np.random.normal(0, 0.09, n), 0.01, 0.99)
    return probs


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    st.markdown(
        f'<h1 style="margin-bottom:4px;">🏎️ F1 Analytics AI Platform'
        f'<span class="version-tag">VERSION {PROJECT_VERSION}</span></h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color:#6e7a94;margin-top:0;">Predictive race analytics · '
        'Ergast + FastF1 · XGBoost walk-forward validated · SHAP-explained</p>',
        unsafe_allow_html=True
    )

    df_raw       = load_data()
    preprocessor = load_preprocessor()

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/640px-F1.svg.png",
            width=110
        )
        st.markdown("""
        <div style="background:rgba(46,204,113,0.1);border:1px solid #2ecc71;border-radius:10px;
                    padding:12px 14px;margin:8px 0 4px 0;">
            <div style="font-size:11px;color:#6e7a94;font-weight:600;letter-spacing:1px;margin-bottom:4px;">ACTIVE MODEL</div>
            <div style="color:#2ecc71;font-size:15px;font-weight:800;">V4 Production</div>
            <div style="color:#6e7a94;font-size:11px;margin-top:2px;">XGBoost · 19 features · Ergast-only</div>
            <div style="margin-top:8px;">
                <span style="background:#2ecc71;color:#080810;font-size:10px;font-weight:800;
                             padding:2px 8px;border-radius:20px;">&#x2713; DEPLOYED</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:4px;font-size:11px;color:#6e7a94;'>See <b>Model Evolution</b> tab for model details and comparison.</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Data Sources**")
        st.markdown("🟢 **Ergast CSV** — Historical (1994–2024)")
        st.markdown("🔵 **FastF1 API** — Practice / Quali / Weather")
        st.markdown("---")
        st.markdown("**V4 Features**")
        st.markdown("""
- Grid, age, experience, form  
- Bayesian circuit history  
- Circuit overtaking index  
- Driver & team Top-10 rates  
        """)

    # Production model — always loaded, not user-selectable
    PRODUCTION_MODEL_PATH = "models/v4_xgb_final.joblib"
    model = load_model(PRODUCTION_MODEL_PATH)
    explainer = build_shap_explainer(model)

    feature_names = preprocessor["features"]

    # ── Tabs ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏎️ Grid Predictor",
        "⚔️ Head-to-Head",
        "⏳ Historical Replay",
        "🏆 Season Projector",
        "📊 Model Evolution",
    ])

    # ══════════════════════════════════════════════════════
    # TAB 1 — Grid Predictor
    # ══════════════════════════════════════════════════════
    with tab1:
        st.subheader("Race Weekend Grid Predictor")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            weather_sim = st.radio("Weather Override", ["Dry", "Mixed", "Wet"], horizontal=True)
        with col_c2:
            sc_sim = st.radio("Safety Car Probability", ["Low", "Medium", "High"], horizontal=True)

        if weather_sim != "Dry" or sc_sim != "Low":
            st.markdown(f"""
            <div class="disclaimer-box">
            <strong>⚠️ Simulation Active</strong> — Weather (<i>{weather_sim}</i>) and
            Safety Car (<i>{sc_sim}</i>) are heuristic overlays on model inputs.
            </div>""", unsafe_allow_html=True)

        years_list = sorted(df_raw[df_raw["year"] >= 2019]["year"].unique(), reverse=True)
        col_y, col_r = st.columns(2)
        with col_y:
            sel_year = st.selectbox("Season", years_list, key="t1_year")
        with col_r:
            races_list = sorted(df_raw[df_raw["year"] == sel_year]["race_name"].unique())
            sel_race = st.selectbox("Grand Prix", races_list, key="t1_race")

        df_race = df_raw[(df_raw["year"] == sel_year) & (df_raw["race_name"] == sel_race)].copy().reset_index(drop=True)

        if df_race.empty:
            st.error("No driver entries found for this race.")
        else:
            st.caption(f"Grid: **{len(df_race)} drivers** · Model: **V4 Production** (XGBoost, 19 features)")

            # Optional grid swap — use checkbox instead of expander to avoid CSS overlap
            if st.checkbox("🔧 Adjust Starting Grid (Optional)", key="t1_grid_adj"):
                adj_driver = st.selectbox("Driver to reposition", sorted(df_race["driverRef"].unique()))
                old_pos = int(df_race.loc[df_race["driverRef"] == adj_driver, "grid"].values[0])
                new_pos = st.slider("New grid position", 1, 20, value=old_pos)
                if new_pos != old_pos:
                    swap_mask = df_race["grid"] == new_pos
                    if swap_mask.any():
                        df_race.loc[swap_mask, "grid"] = old_pos
                    df_race.loc[df_race["driverRef"] == adj_driver, "grid"] = new_pos
                    st.success(f"Repositioned {adj_driver} → P{new_pos}")


            df_sim   = apply_weather_override(df_race, weather_sim)
            X_model  = process_features(df_sim, preprocessor)  # shape (n, n_features), index 0..n-1
            probs    = model.predict_proba(X_model)[:, 1]
            probs    = apply_sc_noise(probs, sc_sim)

            # Build results — keep positional index aligned with X_model
            df_result = df_sim.copy()
            df_result["pred_prob"] = probs
            df_result = df_result.sort_values("pred_prob", ascending=False).reset_index(drop=True)
            df_result["pred_rank"] = df_result.index + 1
            pts_map = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
            df_result["pred_pts"] = df_result["pred_rank"].map(pts_map).fillna(0).astype(int)

            # Re-compute X_model in sorted order so iloc[i] aligns with df_result row i
            X_model_sorted = process_features(df_result, preprocessor)

            st.markdown("### 🏁 Predicted Leaderboard")

            # ── Clean leaderboard table ──────────────────────
            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
            display_rows = []
            for _, row in df_result.iterrows():
                rank = int(row["pred_rank"])
                display_rows.append({
                    "Pos": medals.get(rank, f"P{rank}"),
                    "Driver": row["driverRef"].replace("_", " ").title(),
                    "Team": row["constructor_name"],
                    "Grid": f"P{int(row['grid'])}",
                    "Top-10 Prob": f"{float(row['pred_prob']):.1%}",
                    "Pts": int(row["pred_pts"]) if rank <= 10 else 0,
                })
            st.dataframe(
                pd.DataFrame(display_rows),
                use_container_width=True,
                hide_index=True,
                height=min(36 * len(display_rows) + 38, 600),
            )

            # ── Driver detail panel (SHAP + stats) ──────────
            st.markdown("---")
            st.markdown("#### 🔍 Driver Detail")
            driver_options = df_result["driverRef"].str.replace("_", " ").str.title().tolist()
            sel_driver_detail = st.selectbox(
                "Select driver for feature breakdown",
                driver_options,
                key="t1_driver_detail"
            )
            detail_raw_ref = df_result.iloc[driver_options.index(sel_driver_detail)]["driverRef"]
            detail_idx = df_result[df_result["driverRef"] == detail_raw_ref].index[0]
            detail_row = df_result.loc[detail_idx]

            col_d1, col_d2 = st.columns([1, 1.5])
            with col_d1:
                st.markdown("**Feature Snapshot**")
                st.dataframe(pd.DataFrame({
                    "Feature": [
                        "Grid position", "Recent form (3R avg)", "Team Top-10 rate",
                        "Career starts", "Circuit overtaking index", "Smoothed circuit avg finish"
                    ],
                    "Value": [
                        f"P{int(detail_row['grid'])}",
                        f"{detail_row['rolling_avg_finish_3']:.1f}",
                        f"{detail_row['constructor_top10_rate']:.1%}",
                        str(int(detail_row['driver_experience'])),
                        f"{detail_row.get('circuit_grid_finish_corr', 0.5):.2f}",
                        f"{detail_row.get('smoothed_circuit_avg_finish', 10.0):.1f}",
                    ]
                }), hide_index=True, use_container_width=True)
            with col_d2:
                st.markdown("**SHAP Feature Contribution**")
                row_array = X_model_sorted.iloc[driver_options.index(sel_driver_detail)].values.reshape(1, -1)
                plot_shap_waterfall(explainer, row_array, feature_names, sel_driver_detail)

    # ══════════════════════════════════════════════════════
    # TAB 2 — Driver Comparison
    # ══════════════════════════════════════════════════════
    with tab2:
        st.subheader("⚔️ Driver Comparison")
        st.caption(
            "Stats reflect each driver's cumulative career record up to their last race in the dataset. "
            "Rolling form (3R avg) reflects their last 3 races. "
            "Circuit history is Bayesian-smoothed from all prior visits to the selected track. "
            "Season points reflect cumulative team points before their last race of the season."
        )

        # ── Display name → driverRef mapping ─────────────────
        # Ergast driverRef uses underscores for multi-word names (max_verstappen)
        # and the father (verstappen = Jos) differs from max_verstappen.
        # This mapping converts human-readable names to correct driverRefs.
        DISPLAY_NAME_MAP = {
            # Current grid (2024)
            "Max Verstappen":    "max_verstappen",
            "Lando Norris":      "norris",
            "Charles Leclerc":   "leclerc",
            "Carlos Sainz":      "sainz",
            "Lewis Hamilton":    "hamilton",
            "George Russell":    "russell",
            "Fernando Alonso":   "alonso",
            "Oscar Piastri":     "piastri",
            "Sergio Perez":      "perez",
            "Lance Stroll":      "stroll",
            "Esteban Ocon":      "ocon",
            "Pierre Gasly":      "gasly",
            "Yuki Tsunoda":      "tsunoda",
            "Nico Hulkenberg":   "hulkenberg",
            "Kevin Magnussen":   "kevin_magnussen",
            "Valtteri Bottas":   "bottas",
            "Zhou Guanyu":       "zhou",
            "Alex Albon":        "albon",
            "Logan Sargeant":    "sargeant",
            "Daniel Ricciardo":  "ricciardo",
            "Liam Lawson":       "lawson",
            "Franco Colapinto":  "colapinto",
            "Oliver Bearman":    "bearman",
            "Jack Doohan":       "doohan",
            # Legends
            "Michael Schumacher": "michael_schumacher",
            "Ayrton Senna":       "senna",
            "Kimi Raikkonen":     "raikkonen",
            "Jenson Button":      "button",
            "Nico Rosberg":       "rosberg",
            "Sebastian Vettel":   "vettel",
            "Rubens Barrichello": "barrichello",
            "David Coulthard":    "coulthard",
            "Mika Hakkinen":      "hakkinen",
            "Jos Verstappen":     "verstappen",
        }

        # Build the reverse map and the sorted display list
        # Only show drivers that actually exist in the dataset
        available_refs = set(df_raw["driverRef"].unique())
        valid_display = {
            name: ref for name, ref in DISPLAY_NAME_MAP.items()
            if ref in available_refs
        }
        # Add any drivers in the dataset not in the map (auto-format their name)
        mapped_refs = set(valid_display.values())
        for ref in sorted(available_refs - mapped_refs):
            display = ref.replace("_", " ").title()
            valid_display[display] = ref

        sorted_display_names = sorted(valid_display.keys())

        col_ha, col_hb = st.columns(2)
        with col_ha:
            name_a = st.selectbox("Driver A", sorted_display_names,
                                  index=sorted_display_names.index("Max Verstappen")
                                  if "Max Verstappen" in sorted_display_names else 0,
                                  key="h2h_driver_a")
        with col_hb:
            name_b = st.selectbox("Driver B", sorted_display_names,
                                  index=sorted_display_names.index("Lando Norris")
                                  if "Lando Norris" in sorted_display_names else 1,
                                  key="h2h_driver_b")

        driver_a = valid_display[name_a]
        driver_b = valid_display[name_b]

        if driver_a == driver_b:
            st.warning("Select two different drivers.")
        else:
            # Use most recent data row for each driver
            d_a = df_raw[df_raw["driverRef"] == driver_a].iloc[-1]
            d_b = df_raw[df_raw["driverRef"] == driver_b].iloc[-1]

            col_ga, col_gb, col_gc = st.columns(3)
            with col_ga:
                grid_a = st.slider(f"{name_a.split()[-1]} Grid", 1, 20, 1, key="h2h_grid_a")
            with col_gb:
                grid_b = st.slider(f"{name_b.split()[-1]} Grid", 1, 20, 3, key="h2h_grid_b")
            with col_gc:
                h2h_circuit = st.selectbox("Circuit", [
                    "Neutral (Monza)", "Monaco", "Silverstone", "Spa", "Singapore", "Bahrain"
                ], index=0, key="h2h_circuit")

            # Circuit properties: (circuitRef, lat, lng, alt, overtaking_index)
            circuit_map = {
                "Neutral (Monza)":  ("monza",      44.34,   9.28,  37, 0.52),
                "Monaco":           ("monaco",      43.73,   7.42,   7, 0.88),
                "Silverstone":      ("silverstone", 52.07,  -1.02, 126, 0.55),
                "Spa":              ("spa",         50.44,   5.97, 401, 0.45),
                "Singapore":        ("singapore",    1.29, 103.86,   0, 0.82),
                "Bahrain":          ("bahrain",     26.03,  50.51,   7, 0.58),
            }
            cref, clat, clng, calt, c_oi = circuit_map[h2h_circuit]

            def make_h2h_row(d_stats, grid, driver_ref):
                """Build one inference row, using the correct circuit-specific history."""
                circ_hist = df_raw[
                    (df_raw["driverRef"] == driver_ref) &
                    (df_raw["circuitRef"] == cref)
                ]
                smoothed_circuit = (
                    float(circ_hist.iloc[-1]["smoothed_circuit_avg_finish"])
                    if not circ_hist.empty else 10.0
                )
                return pd.DataFrame([{
                    "grid":                       grid,
                    "driver_age":                 d_stats["driver_age"],
                    "driver_experience":          d_stats["driver_experience"],
                    "driver_top10_rate":          d_stats["driver_top10_rate"],
                    "constructor_top10_rate":     d_stats["constructor_top10_rate"],
                    "rolling_avg_finish_3":       d_stats["rolling_avg_finish_3"],
                    "rolling_avg_finish_5":       d_stats["rolling_avg_finish_5"],
                    "prev_race_finish":           d_stats["prev_race_finish"],
                    "home_race":                  0,
                    "grid_qualifying_diff":       0.0,
                    "constructor_season_points":  d_stats["constructor_season_points"],
                    "lat": clat, "lng": clng, "alt": calt,
                    "smoothed_circuit_avg_finish": smoothed_circuit,
                    "circuit_grid_finish_corr":    c_oi,
                    "driverRef":        driver_ref,
                    "constructor_name": d_stats["constructor_name"],
                    "circuitRef":       cref,
                }])

            row_a = make_h2h_row(d_a, grid_a, driver_a)
            row_b = make_h2h_row(d_b, grid_b, driver_b)

            Xa = process_features(row_a, preprocessor)
            Xb = process_features(row_b, preprocessor)

            p_a = float(model.predict_proba(Xa)[0][1])
            p_b = float(model.predict_proba(Xb)[0][1])

            st.markdown("---")
            col_r1, col_r2, col_r3 = st.columns([2, 1, 2])
            with col_r1:
                st.metric(label=f"{name_a.upper()} — Top-10 probability",
                          value=f"{p_a:.1%}")
            with col_r2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("### VS")
            with col_r3:
                st.metric(label=f"{name_b.upper()} — Top-10 probability",
                          value=f"{p_b:.1%}")

            st.info(
                f"**{h2h_circuit}** — Circuit Overtaking Index: {c_oi:.2f} "
                f"({'grid position dominates' if c_oi > 0.75 else 'normal overtaking' if c_oi > 0.55 else 'high overtaking circuit'})"
            )

            st.caption(
                "**Note on Probabilities:** The production model is an XGBoost binary classifier trained to predict "
                "each driver's independent probability of finishing in the points (Top 10). It is *not* a pairwise ranking model. "
                "Pairwise finishing order prediction (e.g., XGBoost Ranker) is a planned future enhancement."
            )

            # ── Performance Comparison table ──────────────────────
            st.markdown("---")
            st.markdown("#### Feature Comparison")
            st.caption(
                "Every value shown here is exactly what enters the XGBoost model. "
                "Source: Career = all races to date | Season = current season cumulative | "
                "Form = last 3 races | Circuit = Bayesian-smoothed prior visits."
            )

            # Extract the smoothed circuit values actually used in inference
            circ_a_val = float(row_a["smoothed_circuit_avg_finish"].values[0])
            circ_b_val = float(row_b["smoothed_circuit_avg_finish"].values[0])

            compare_df = pd.DataFrame({
                "Feature": [
                    "Grid Position",
                    "Career Starts (Experience)",
                    "Driver Top-10 Rate",
                    "Team Top-10 Rate",
                    "Recent Form — 3-race avg finish",
                    "Team Season Points (cumulative)",
                    f"Circuit Avg Finish (Bayesian, {h2h_circuit})",
                    "Circuit Overtaking Index",
                    "Constructor",
                    "Data up to",
                ],
                "Source": [
                    "User input", "Career", "Career", "Career",
                    "Last 3 races", "Season (before last race)",
                    "Circuit-specific", "Circuit-specific",
                    "Season", "Season",
                ],
                name_a: [
                    f"P{grid_a}",
                    int(d_a["driver_experience"]),
                    f"{d_a['driver_top10_rate']:.1%}",
                    f"{d_a['constructor_top10_rate']:.1%}",
                    f"{d_a['rolling_avg_finish_3']:.2f}",
                    f"{d_a['constructor_season_points']:.0f} pts",
                    f"{circ_a_val:.2f}",
                    f"{c_oi:.2f}",
                    d_a["constructor_name"],
                    f"{int(d_a['year'])} R{int(d_a['round'])}",
                ],
                name_b: [
                    f"P{grid_b}",
                    int(d_b["driver_experience"]),
                    f"{d_b['driver_top10_rate']:.1%}",
                    f"{d_b['constructor_top10_rate']:.1%}",
                    f"{d_b['rolling_avg_finish_3']:.2f}",
                    f"{d_b['constructor_season_points']:.0f} pts",
                    f"{circ_b_val:.2f}",
                    f"{c_oi:.2f}",
                    d_b["constructor_name"],
                    f"{int(d_b['year'])} R{int(d_b['round'])}",
                ],
            })
            st.dataframe(compare_df, hide_index=True, use_container_width=True)

            # ── Key Advantages ────────────────────────────────────
            st.markdown("#### Key Advantages")
            for line in [
                (f"✅ **{name_a}** has grid advantage (P{grid_a} vs P{grid_b})"
                 if grid_a < grid_b else
                 f"✅ **{name_b}** has grid advantage (P{grid_b} vs P{grid_a})"),
                (f"✅ **{name_a}** has better recent form (avg P{d_a['rolling_avg_finish_3']:.1f} vs P{d_b['rolling_avg_finish_3']:.1f})"
                 if d_a["rolling_avg_finish_3"] < d_b["rolling_avg_finish_3"] else
                 f"✅ **{name_b}** has better recent form (avg P{d_b['rolling_avg_finish_3']:.1f} vs P{d_a['rolling_avg_finish_3']:.1f})"),
                (f"✅ **{name_a}** has stronger constructor: {d_a['constructor_name']} "
                 f"(Top-10 rate {d_a['constructor_top10_rate']:.1%})"
                 if d_a["constructor_top10_rate"] > d_b["constructor_top10_rate"] else
                 f"✅ **{name_b}** has stronger constructor: {d_b['constructor_name']} "
                 f"(Top-10 rate {d_b['constructor_top10_rate']:.1%})"),
            ]:
                st.markdown(line)

            # ── SHAP Waterfall for each driver ────────────────────
            st.markdown("---")
            st.markdown("#### SHAP — Why did the model assign these probabilities?")
            st.caption(
                "SHAP values show which features push each driver's prediction up or down "
                "relative to the model's average prediction. Features in red increase Top-10 probability."
            )
            col_sh1, col_sh2 = st.columns(2)
            with col_sh1:
                st.markdown(f"**{name_a}** — P(Top 10) = {p_a:.1%}")
                plot_shap_waterfall(explainer, Xa.values.reshape(1, -1), feature_names, name_a)
            with col_sh2:
                st.markdown(f"**{name_b}** — P(Top 10) = {p_b:.1%}")
                plot_shap_waterfall(explainer, Xb.values.reshape(1, -1), feature_names, name_b)

    # ══════════════════════════════════════════════════════
    # TAB 3 — Historical Replay
    # ══════════════════════════════════════════════════════
    with tab3:
        st.subheader("⏳ Historical Race Replay")
        st.caption("Run the model on a past race and compare predictions vs. actual results.")

        col_ry, col_rr = st.columns(2)
        with col_ry:
            hist_years = sorted(df_raw["year"].unique(), reverse=True)
            replay_year = st.selectbox("Season", hist_years, index=1, key="rep_year")
        with col_rr:
            hist_races = sorted(df_raw[df_raw["year"] == replay_year]["race_name"].unique())
            replay_race = st.selectbox("Grand Prix", hist_races, key="rep_race")

        if st.button("▶ Run Replay"):
            df_hist = df_raw[(df_raw["year"] == replay_year) & (df_raw["race_name"] == replay_race)].copy().reset_index(drop=True)

            if df_hist.empty:
                st.error("No data found for this race.")
            else:
                X_hist = process_features(df_hist, preprocessor)
                probs_hist = model.predict_proba(X_hist)[:, 1]
                df_hist["pred_prob"] = probs_hist
                df_hist = df_hist.sort_values("pred_prob", ascending=False).reset_index(drop=True)
                df_hist["pred_rank"] = df_hist.index + 1

                actual_map = df_hist.set_index("driverRef")["Top10"].to_dict()
                df_hist["actual_Top10"] = df_hist["driverRef"].map(actual_map)

                predicted_top10 = set(df_hist.head(10)["driverRef"])
                actual_top10    = set(df_hist[df_hist["Top10"] == 1]["driverRef"])
                correct = len(predicted_top10 & actual_top10)

                col_rp1, col_rp2 = st.columns(2)
                with col_rp1:
                    st.markdown("#### Model Predictions")
                    st.dataframe(
                        df_hist[["pred_rank", "driverRef", "grid", "pred_prob"]]
                        .rename(columns={"pred_rank": "Rank", "driverRef": "Driver",
                                         "grid": "Grid", "pred_prob": "Probability"})
                        .style.format({"Probability": "{:.1%}"})
                    )
                with col_rp2:
                    st.markdown("#### Actual Top 10")
                    act_df = df_hist[df_hist["Top10"] == 1][["driverRef", "constructor_name", "grid"]].reset_index(drop=True)
                    act_df.index += 1
                    st.dataframe(act_df.rename(columns={"driverRef": "Driver", "constructor_name": "Team", "grid": "Grid"}))

                pct = correct / 10
                color = "green" if pct >= 0.7 else "orange" if pct >= 0.5 else "red"
                st.markdown(f"#### Points Finishers Correctly Predicted")
                st.metric(label="Correct out of 10", value=f"{correct}/10")

    # ══════════════════════════════════════════════════════
    # TAB 4 — Season Projector
    # ══════════════════════════════════════════════════════
    with tab4:
        st.subheader("🏆 Season Standing Projections")
        st.caption("Simulate every round of a season and project final standings.")
        st.markdown(f"""
        <div class="disclaimer-box">
        <strong>⚠️ Projection Disclaimer</strong> — The model predicts <i>probability of a Top 10 finish</i>, 
        not exact race pace. The points assigned here are based on the probability ranking for each race, 
        which over-rewards consistent midfield finishers and under-rewards sporadic winners.
        </div>""", unsafe_allow_html=True)

        sim_year = st.selectbox("Season", [2024, 2023, 2022, 2021, 2020, 2019])

        if st.button("▶ Run Full Season Simulation"):
            df_season = df_raw[df_raw["year"] == sim_year].copy()
            if df_season.empty:
                st.error("No data for selected season.")
            else:
                rounds = sorted(df_season["round"].unique())
                driver_pts = {}
                constr_pts = {}
                pts_map = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}

                progress = st.progress(0.0)
                for i, r in enumerate(rounds):
                    df_r = df_season[df_season["round"] == r].copy().reset_index(drop=True)
                    X_r  = process_features(df_r, preprocessor)
                    probs_r = model.predict_proba(X_r)[:, 1]
                    df_r["pred_prob"] = probs_r
                    df_r = df_r.sort_values("pred_prob", ascending=False).reset_index(drop=True)
                    for idx, row in df_r.iterrows():
                        pts = pts_map.get(idx + 1, 0)
                        driver_pts[row["driverRef"]] = driver_pts.get(row["driverRef"], 0) + pts
                        constr_pts[row["constructor_name"]] = constr_pts.get(row["constructor_name"], 0) + pts
                    progress.progress((i + 1) / len(rounds))

                sd = sorted(driver_pts.items(), key=lambda x: x[1], reverse=True)
                sc = sorted(constr_pts.items(), key=lambda x: x[1], reverse=True)

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown("#### Driver Championship")
                    df_d = pd.DataFrame(sd, columns=["Driver", "Pts"])
                    df_d["Driver"] = df_d["Driver"].str.capitalize()
                    df_d.index += 1
                    st.dataframe(df_d)
                with col_s2:
                    st.markdown("#### Constructor Championship")
                    df_c = pd.DataFrame(sc, columns=["Team", "Pts"])
                    df_c.index += 1
                    st.dataframe(df_c)

                st.success(f"🏆 Projected {sim_year} Champion: **{sd[0][0].capitalize()}** ({sd[0][1]} pts)")

    # ══════════════════════════════════════════════════════
    # TAB 5 — Model Evolution
    # ══════════════════════════════════════════════════════
    with tab5:
        st.subheader("📊 Model Evolution — V3 Baseline to V4 Production")
        st.caption("All metrics computed via walk-forward validation on held-out seasons 2019–2024. "
                   "The model is never evaluated on data it was trained on.")

        phase_metrics = load_phase_metrics()

        if not phase_metrics:
            st.info("Run `notebooks/v4_training_pipeline.py` to generate phase metrics.")
        else:
            # ── Model Comparison Table ─────────────────────────────────
            st.markdown("#### Model Comparison")

            PHASE_META = {
                "V3 Baseline": {
                    "features_added": "Grid, age, experience, Top-10 rates, rolling form, home race, circuit location",
                    "status": "Baseline",
                    "status_color": "#6e7a94",
                    "n_features": 17,
                },
                "Phase 2 (Circuit Hist.)": {
                    "features_added": "+ Bayesian circuit avg finish, circuit overtaking index",
                    "status": "✅ Production",
                    "status_color": "#2ecc71",
                    "n_features": 19,
                },
                "Phase 5 (Champ. Form)": {
                    "features_added": "+ Championship points/position, 5-race rolling form (driver & team)",
                    "status": "Experimental",
                    "status_color": "#f0a500",
                    "n_features": 25,
                },
                "Phase 3 (Weather+SC)": {
                    "features_added": "+ Historical safety car rate (requires FastF1)",
                    "status": "Experimental",
                    "status_color": "#f0a500",
                    "n_features": 26,
                },
            }

            metrics_keys = ["accuracy", "precision", "recall", "f1", "auc"]
            table_rows = []
            for phase_label, m in phase_metrics.items():
                meta = PHASE_META.get(phase_label, {})
                table_rows.append({
                    "Model": phase_label,
                    "Features Added": meta.get("features_added", "—"),
                    "# Features": meta.get("n_features", "—"),
                    "Accuracy":  round(m.get("accuracy",  0), 4),
                    "Precision": round(m.get("precision", 0), 4),
                    "Recall":    round(m.get("recall",    0), 4),
                    "F1":        round(m.get("f1",        0), 4),
                    "ROC-AUC":   round(m.get("auc",       0), 4),
                    "Status":    meta.get("status", "—"),
                })

            comparison_df = pd.DataFrame(table_rows).set_index("Model")

            def style_table(df):
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
                    if col in df.columns:
                        best_val = df[col].max()
                        for idx in df.index:
                            if df.loc[idx, col] == best_val:
                                styles.loc[idx, col] = "color: #2ecc71; font-weight: 800"
                for idx in df.index:
                    if "Production" in str(df.loc[idx, "Status"]):
                        styles.loc[idx, "Status"] = "color: #2ecc71; font-weight: 800"
                    elif "Experimental" in str(df.loc[idx, "Status"]):
                        styles.loc[idx, "Status"] = "color: #f0a500;"
                    elif "Baseline" in str(df.loc[idx, "Status"]):
                        styles.loc[idx, "Status"] = "color: #6e7a94;"
                return styles

            st.dataframe(
                comparison_df.style.apply(style_table, axis=None).format(
                    {"Accuracy": "{:.4f}", "Precision": "{:.4f}",
                     "Recall": "{:.4f}", "F1": "{:.4f}", "ROC-AUC": "{:.4f}"}
                ),
                use_container_width=True,
                height=220,
            )

            # ── Production Rationale ───────────────────────────────────
            st.markdown("---")
            st.markdown("#### Why V4 Phase 2 is the Production Model")
            st.markdown("""
            <div style="background:rgba(46,204,113,0.07);border-left:4px solid #2ecc71;
                        padding:14px 18px;border-radius:8px;margin-bottom:8px;">
            <b style="color:#2ecc71;">&#x2713; Selected for production: Phase 2 (Circuit History)</b>
            <ul style="margin:10px 0 0 0;color:#c8d0de;font-size:14px;line-height:1.8;">
                <li><b>Reproducibility:</b> Uses only Ergast CSV data — no FastF1 API dependency, no internet required at inference time.</li>
                <li><b>Competitive metrics:</b> ROC-AUC 0.8367 vs V3 baseline 0.8372 — marginal drop offset by significantly better circuit-specific context.</li>
                <li><b>Bayesian circuit history:</b> Provides the model with track-specific driver tendencies while gracefully handling sparse data via the global career prior.</li>
                <li><b>Minimal complexity:</b> 19 features (vs 25 in Phase 5) — easier to interpret, maintain, and debug in production.</li>
                <li><b>Zero leakage:</b> All 19 features verified as strictly pre-race by the automated leakage checker.</li>
            </ul>
            </div>
            <div style="background:rgba(240,165,0,0.07);border-left:4px solid #f0a500;
                        padding:14px 18px;border-radius:8px;">
            <b style="color:#f0a500;">Why Phase 5 and Phase 3 are kept experimental</b>
            <ul style="margin:10px 0 0 0;color:#c8d0de;font-size:14px;line-height:1.8;">
                <li><b>Phase 5</b> adds championship standings — these improve AUC (+0.0027) but require driver_standings.csv to be kept perfectly up-to-date mid-season.</li>
                <li><b>Phase 3</b> requires FastF1 for historical safety car rates — API calls introduce latency and potential failures in a production environment.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            # ── Per-Year Walk-Forward Chart ────────────────────────────
            st.markdown("---")
            st.markdown("#### Walk-Forward Validation — Production Model (Per Season)")
            prod_key = "Phase 2 (Circuit Hist.)"
            if prod_key in phase_metrics and "per_year" in phase_metrics[prod_key]:
                per_yr_df = pd.DataFrame(phase_metrics[prod_key]["per_year"]).set_index("year")
                fig_yr, ax = plt.subplots(figsize=(10, 3.5))
                fig_yr.patch.set_facecolor("#13141f")
                ax.set_facecolor("#13141f")
                for metric, color, lw in [("auc", "#ff1801", 2.5), ("f1", "#00b2ff", 2), ("accuracy", "#2ecc71", 2)]:
                    if metric in per_yr_df.columns:
                        ax.plot(per_yr_df.index, per_yr_df[metric], marker="o",
                                label=metric.upper(), color=color, linewidth=lw)
                ax.axhline(0.77, color="#3a3b50", linestyle="--", linewidth=1, label="Avg accuracy baseline")
                ax.tick_params(colors="#e0e6ed")
                ax.set_xlabel("Season", color="#6e7a94")
                ax.set_ylabel("Score", color="#6e7a94")
                ax.set_title("Production Model — Walk-Forward Metrics by Season (2019–2024)",
                             color="#e0e6ed", fontsize=11)
                ax.legend(facecolor="#13141f", labelcolor="#e0e6ed", fontsize=9)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1e2035")
                plt.tight_layout()
                st.pyplot(fig_yr)
                plt.close(fig_yr)
            else:
                st.info("Per-year data available after running `notebooks/v4_training_pipeline.py --phase 2`.")

            # ── SHAP Feature Importance ────────────────────────────────
            st.markdown("---")
            st.markdown("#### SHAP Feature Importance")
            st.caption("SHAP beeswarm plots show how each feature pushes predictions higher or lower. "
                       "Points to the right = increased Top-10 probability. Red = high feature value.")
            shap_phases = {
                "V3 Baseline":               "reports/v3_baseline_shap_beeswarm.png",
                "Phase 2 — Production":       "reports/v4_phase2_shap_beeswarm.png",
                "Phase 5 — Experimental":     "reports/v4_phase5_shap_beeswarm.png",
                "Phase 3 — Experimental":     "reports/v4_phase3_shap_beeswarm.png",
            }
            available_shaps = {k: v for k, v in shap_phases.items() if os.path.exists(v)}
            if available_shaps:
                shap_sel = st.selectbox("Select model phase", list(available_shaps.keys()),
                                        key="shap_phase_sel")
                st.image(available_shaps[shap_sel], use_container_width=True)


if __name__ == "__main__":
    main()
