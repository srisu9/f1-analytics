import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

                                                                                
st.set_page_config(
    page_title="F1 Analytics AI Platform",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

                                                       
st.markdown("""
<style>
    .stApp { background-color: #0e0f12; color: #e0e6ed; }
    h1, h2, h3, h4, h5 {
        color: #ff1801 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 800;
    }
    section[data-testid="stSidebar"] {
        background-color: #15161e;
        border-right: 1px solid #222530;
    }
    .metric-card {
        background-color: #1a1c24;
        border-left: 5px solid #ff1801;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #ffffff; }
    .metric-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .disclaimer-box {
        background-color: #2b1818;
        border-left: 4px solid #ff1801;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 13px;
        color: #fca5a5;
    }
    /* Style tables nicely */
    .grid-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 14px;
        text-align: left;
    }
    .grid-table th {
        background-color: #1a1c24;
        color: #ff1801;
        font-weight: bold;
        padding: 10px;
        border-bottom: 2px solid #222530;
    }
    .grid-table td {
        padding: 10px;
        border-bottom: 1px solid #222530;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Loads the main pre-engineered dataset."""
    df = pd.read_csv("data/processed/model_ready.csv")
    df_sorted = df.sort_values("date")                                       
    return df_sorted


@st.cache_resource
def load_preprocessor():
    """Loads preprocessor joblib containing scaler and target encoders."""
    return joblib.load("models/preprocessor.joblib")


@st.cache_resource
def load_xgboost():
    """Loads Tuned XGBoost (Default Model)."""
    return joblib.load("models/xgboost_wfv.joblib")


@st.cache_resource
def load_rf():
    """Loads Random Forest model."""
    return joblib.load("models/random_forest.joblib")


@st.cache_resource
def load_ensemble():
    """Loads the soft-voting ensemble predictor from src/ensemble.py."""
    from src.ensemble import EnsemblePredictor
    return EnsemblePredictor()


@st.cache_resource
def build_shap_explainer(_model):
    """Build TreeExplainer for SHAP explanations."""
    try:
                                          
        return shap.TreeExplainer(_model)
    except Exception:
        return None


def apply_simulation_heuristics(df, weather, safety_car):
    """
    Applies simulation heuristics to reflect changed race conditions in inputs.
    Note: These are simulation rules and not directly learned by XGBoost.
    """
    df_sim = df.copy()

                    
    if weather == "Mixed":
                                                                             
        df_sim["grid"] = 10.5 + 0.7 * (df_sim["grid"] - 10.5)
                                              
        df_sim["driver_experience"] = df_sim["driver_experience"] * 1.10
        df_sim["driver_win_rate"] = df_sim["driver_win_rate"] * 1.10
    elif weather == "Wet":
                                                                             
        df_sim["grid"] = 10.5 + 0.45 * (df_sim["grid"] - 10.5)
                                                                        
        df_sim["driver_experience"] = df_sim["driver_experience"] * 1.25
        df_sim["driver_win_rate"] = df_sim["driver_win_rate"] * 1.20
                                                                        
        df_sim["rolling_avg_finish_3"] = df_sim["rolling_avg_finish_3"] + 1.0

    return df_sim


def apply_safety_car_noise(probs, safety_car_level):
    """Applies random chance noise to simulate safety car disruptions."""
    np.random.seed(42)                                            
    n = len(probs)
    
    if safety_car_level == "Medium":
        noise = np.random.normal(0, 0.04, n)
        probs = np.clip(probs + noise, 0.01, 0.99)
    elif safety_car_level == "High":
        noise = np.random.normal(0, 0.09, n)
        probs = np.clip(probs + noise, 0.01, 0.99)
        
                                                                                              
    return probs


def process_features(df_race, preprocessor):
    """Transforms raw race features using preprocessor encoders and scalers."""
    enc = preprocessor["encoders"]
    features_to_scale = preprocessor["features"]
    
    df_processed = df_race.copy()
    
                                        
    for cat in ["driverRef", "constructor_name", "circuitRef"]:
        smooth_map = enc[cat]["map"]
        global_mean = enc[cat]["global_mean"]
        df_processed[cat + "_encoded"] = df_processed[cat].map(smooth_map).fillna(global_mean)

                                                            
    return df_processed[features_to_scale]


def plot_shap_waterfall(explainer, input_row, features_list, driver_name):
    """Generates a SHAP waterfall plot and renders in Streamlit."""
    if explainer is None:
        st.info("SHAP details not supported for current model selection.")
        return
        
    try:
        shap_values = explainer(input_row)
        
                                             
        sv = shap_values
        if hasattr(sv, "values") and sv.values.ndim == 3:
            sv = shap.Explanation(
                values=sv.values[0, :, 1],
                base_values=sv.base_values[0, 1] if sv.base_values.ndim > 1 else sv.base_values[0],
                data=sv.data[0],
                feature_names=features_list
            )
        elif hasattr(sv, "values") and sv.values.ndim == 2:
            sv = shap.Explanation(
                values=sv.values[0],
                base_values=sv.base_values[0] if hasattr(sv.base_values, "__len__") else sv.base_values,
                data=sv.data[0],
                feature_names=features_list
            )

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#15161e")
        ax.set_facecolor("#15161e")
        
        shap.plots.waterfall(sv, max_display=8, show=False)
        
                              
        for a in fig.get_axes():
            a.set_facecolor("#15161e")
            a.tick_params(colors="#e0e6ed", labelsize=8)
            a.xaxis.label.set_color("#e0e6ed")
            a.yaxis.label.set_color("#e0e6ed")
            a.title.set_color("#e0e6ed")
            for spine in a.spines.values():
                spine.set_edgecolor("#222530")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as exc:
        st.warning(f"Could not render SHAP plot for {driver_name}: {exc}")


def main():
    st.title("🏎️ F1 Analytics AI Platform (Version 3)")
    st.write("An end-to-end predictive platform for entire F1 Grand Prix grids, comparisons, and simulations.")

                    
    df_raw = load_data()
    preprocessor = load_preprocessor()
    
                                                                                
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/640px-F1.svg.png",
        width=120
    )
    st.sidebar.markdown("### Model Configuration")
    
    model_choice = st.sidebar.selectbox(
        "Active Prediction Model",
        ["Tuned XGBoost (Default)", "Random Forest", "Soft-Voting Ensemble"],
        index=0
    )
    
                       
    if model_choice == "Tuned XGBoost (Default)":
        model = load_xgboost()
        explainer_model = model
    elif model_choice == "Random Forest":
        model = load_rf()
        explainer_model = model
    else:
        model = load_ensemble()
                                                                     
        explainer_model = load_xgboost()

    explainer = build_shap_explainer(explainer_model)

                                                                                
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏎️ Grid Predictor & Simulator",
        "⚔️ Driver Head-to-Head",
        "⏳ Historical Replays",
        "🏆 Season Predictor"
    ])

                                                                                
    with tab1:
        st.subheader("Race Weekend Simulator")
        
                                  
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            weather_sim = st.radio("Simulation Weather Conditions", ["Dry", "Mixed", "Wet"], index=0, horizontal=True)
        with col_ctrl2:
            safety_car_sim = st.radio("Safety Car Probability", ["Low", "Medium", "High"], index=0, horizontal=True)
            
        if weather_sim != "Dry" or safety_car_sim != "Low":
            st.markdown(f"""
            <div class="disclaimer-box">
                <strong>⚠️ Simulation Heuristic Active:</strong> Weather (<i>{weather_sim}</i>) and Safety Car (<i>{safety_car_sim}</i>) 
                features are simulation overlays adjusting the core ML inputs (e.g. scaling grid dominance and boosting experience metrics). 
                They are not directly learned by the model from historical weather records.
            </div>
            """, unsafe_allow_html=True)
            
                     
        years_list = sorted(list(df_raw[df_raw["year"] >= 2019]["year"].unique()), reverse=True)
        sel_year = st.selectbox("Select Season", years_list, key="t1_year")
        
        races_list = sorted(list(df_raw[df_raw["year"] == sel_year]["race_name"].unique()))
        sel_race = st.selectbox("Select Grand Prix", races_list, key="t1_race")
        
                           
        df_race = df_raw[(df_raw["year"] == sel_year) & (df_raw["race_name"] == sel_race)].copy()
        
        if df_race.empty:
            st.error("No driver entries found for this race.")
        else:
            st.write(f"Loaded **{len(df_race)}** drivers from the qualifying grid.")
            
                                                       
            st.markdown("#### Adjust Driver Starting Grid Positions (Optional)")
            adjust_driver = st.selectbox("Select Driver to Swap Position", sorted(df_race["driverRef"].unique()))
            old_grid = int(df_race[df_race["driverRef"] == adjust_driver]["grid"].values[0])
            new_grid = st.slider("New Grid Position", 1, 20, value=old_grid)
            
            if new_grid != old_grid:
                                                                 
                swap_driver = df_race[df_race["grid"] == new_grid]["driverRef"].values
                if len(swap_driver) > 0:
                    df_race.loc[df_race["driverRef"] == swap_driver[0], "grid"] = old_grid
                df_race.loc[df_race["driverRef"] == adjust_driver, "grid"] = new_grid
                st.info(f"Swapped starting grid positions: {adjust_driver} to P{new_grid}")

                              
            df_simulated = apply_simulation_heuristics(df_race, weather_sim, safety_car_sim)
            
                     
            X_model = process_features(df_simulated, preprocessor)
            probs = model.predict_proba(X_model)[:, 1]
            probs_sim = apply_safety_car_noise(probs, safety_car_sim)
            
            df_race["pred_prob"] = probs_sim
            df_race = df_race.sort_values("pred_prob", ascending=False)
            df_race["pred_rank"] = np.arange(1, len(df_race) + 1)
            
                                     
            points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
            df_race["pred_points"] = df_race["pred_rank"].map(points_map).fillna(0).astype(int)

                                
            st.markdown("### 🏁 Predicted Race Weekend Leaderboard")
            
            for idx, row in df_race.iterrows():
                rank = row["pred_rank"]
                driver = row["driverRef"].capitalize()
                team = row["constructor_name"]
                grid = int(row["grid"])
                prob = row["pred_prob"]
                pts = row["pred_points"]
                
                                        
                gauge_color = "#ff1801" if prob < 0.5 else "#00b2ff" if prob < 0.8 else "#2ecc71"
                
                exp_title = f"{rank}. {driver} ({team}) — Starting Grid: P{grid} | Predicted Probability: {prob:.1%}"
                if rank <= 10:
                    exp_title += f" | Points: {pts} pts"
                
                with st.expander(exp_title):
                    col_det1, col_det2 = st.columns([1, 1.5])
                    with col_det1:
                        st.markdown("##### Feature Strengths")
                        st.markdown(f"""
                        * **Qualifying Grid**: Qualified P{grid} 
                        * **Recent Form (3R)**: Average finish position of {row['rolling_avg_finish_3']:.1f}
                        * **Team Performance**: Historical Team Win Rate of {row['constructor_win_rate']:.1%}
                        * **Experience**: {int(row['driver_experience'])} career race starts
                        * **Track Profile**: Circuit elevation of {row['alt']}m
                        """)
                    with col_det2:
                        st.markdown("##### Live Feature Contribution (SHAP)")
                        row_feat = X_model.loc[row.name].values.reshape(1, -1)
                        plot_shap_waterfall(explainer, row_feat, preprocessor["features"], driver)

                                                                                
    with tab2:
        st.subheader("⚔️ Driver Head-to-Head Comparison")
        
                                     
        unique_drivers = sorted(df_raw["driverRef"].unique())
        
        col_h2h1, col_h2h2 = st.columns(2)
        with col_h2h1:
            driver_a = st.selectbox("Driver A", unique_drivers, index=unique_drivers.index("verstappen") if "verstappen" in unique_drivers else 0)
        with col_h2h2:
            driver_b = st.selectbox("Driver B", unique_drivers, index=unique_drivers.index("norris") if "norris" in unique_drivers else 1)
            
        if driver_a == driver_b:
            st.warning("Please select two different drivers for comparison.")
        else:
                                
            d_stats_a = df_raw[df_raw["driverRef"] == driver_a].iloc[-1]
            d_stats_b = df_raw[df_raw["driverRef"] == driver_b].iloc[-1]
            
                                                                        
            col_pos1, col_pos2 = st.columns(2)
            with col_pos1:
                grid_a = st.slider(f"{driver_a.capitalize()} Grid", 1, 20, 2)
            with col_pos2:
                grid_b = st.slider(f"{driver_b.capitalize()} Grid", 1, 20, 3)

                                
            def make_virtual_row(d_stats, grid, opponent_stats):
                return pd.DataFrame([{
                    "grid":                      grid,
                    "driver_age":                d_stats["driver_age"],
                    "driver_experience":         d_stats["driver_experience"],
                    "driver_win_rate":           d_stats["driver_win_rate"],
                    "constructor_win_rate":      d_stats["constructor_win_rate"],
                    "rolling_avg_finish_3":      d_stats["rolling_avg_finish_3"],
                    "rolling_avg_finish_5":      d_stats["rolling_avg_finish_5"],
                    "prev_race_finish":          d_stats["prev_race_finish"],
                    "home_race":                 0,
                    "grid_qualifying_diff":      0.0,
                    "constructor_season_points": d_stats["constructor_season_points"],
                    "lat":                       44.34,             
                    "lng":                       9.28,              
                    "alt":                       183,               
                    "driverRef":                 d_stats["driverRef"],
                    "constructor_name":          d_stats["constructor_name"],
                    "circuitRef":                "monza"
                }])

            row_a = make_virtual_row(d_stats_a, grid_a, d_stats_b)
            row_b = make_virtual_row(d_stats_b, grid_b, d_stats_a)
            
            row_a_proc = process_features(row_a, preprocessor)
            row_b_proc = process_features(row_b, preprocessor)
            
            p_a = float(model.predict_proba(row_a_proc)[0][1])
            p_b = float(model.predict_proba(row_b_proc)[0][1])
            
                                   
            p_a_beats_b = p_a / (p_a + p_b + 1e-9)
            
                                        
            st.markdown("### Head-to-Head Prediction")
            st.markdown(f"""
            <div style="background:#1a1c24;padding:25px;border-radius:12px;text-align:center;box-shadow:0 4px 15px rgba(0,0,0,0.5);">
                <div style="font-size:16px;color:#8b949e;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">
                    Probability that {driver_a.capitalize()} finishes ahead of {driver_b.capitalize()}
                </div>
                <div style="font-size:64px;font-weight:900;color:#ff1801;">
                    {p_a_beats_b:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
                                    
            st.markdown("#### Performance Breakdown")
            stats_compare = pd.DataFrame({
                "Stat": ["Age", "Starts (Experience)", "Driver Historical Win Rate", "Team Historical Win Rate", "Recent Form (3R Avg)"],
                driver_a.capitalize(): [
                    f"{d_stats_a['driver_age']:.1f} yrs",
                    int(d_stats_a['driver_experience']),
                    f"{d_stats_a['driver_win_rate']:.1%}",
                    f"{d_stats_a['constructor_win_rate']:.1%}",
                    f"{d_stats_a['rolling_avg_finish_3']:.1f}"
                ],
                driver_b.capitalize(): [
                    f"{d_stats_b['driver_age']:.1f} yrs",
                    int(d_stats_b['driver_experience']),
                    f"{d_stats_b['driver_win_rate']:.1%}",
                    f"{d_stats_b['constructor_win_rate']:.1%}",
                    f"{d_stats_b['rolling_avg_finish_3']:.1f}"
                ]
            })
            st.table(stats_compare)

                                     
            st.markdown("#### Comparison Insights")
            reasons = []
            if grid_a < grid_b:
                reasons.append(f"✓ **{driver_a.capitalize()}** has starting grid advantage (P{grid_a} vs P{grid_b}).")
            else:
                reasons.append(f"✓ **{driver_b.capitalize()}** has starting grid advantage (P{grid_b} vs P{grid_a}).")
                
            if d_stats_a["rolling_avg_finish_3"] < d_stats_b["rolling_avg_finish_3"]:
                reasons.append(f"✓ **{driver_a.capitalize()}** enters in better recent form (3-race avg: {d_stats_a['rolling_avg_finish_3']:.1f} vs {d_stats_b['rolling_avg_finish_3']:.1f}).")
            else:
                reasons.append(f"✓ **{driver_b.capitalize()}** enters in better recent form (3-race avg: {d_stats_b['rolling_avg_finish_3']:.1f} vs {d_stats_a['rolling_avg_finish_3']:.1f}).")

            if d_stats_a["constructor_win_rate"] > d_stats_b["constructor_win_rate"]:
                reasons.append(f"✓ **{driver_a.capitalize()}** enjoys constructor package advantage ({d_stats_a['constructor_name']} vs {d_stats_b['constructor_name']}).")
            else:
                reasons.append(f"✓ **{driver_b.capitalize()}** enjoys constructor package advantage ({d_stats_b['constructor_name']} vs {d_stats_a['constructor_name']}).")
                
            for r in reasons:
                st.markdown(r)

                                                                                
    with tab3:
        st.subheader("⏳ Historical Race Replays")
        st.write("Evaluate the model by loading historical race configurations, predicting, and comparing to actual outcomes.")

                       
        hist_years = sorted(list(df_raw["year"].unique()), reverse=True)
        replay_year = st.selectbox("Season Year", hist_years, index=1, key="rep_year")
        
                       
        hist_races = sorted(list(df_raw[df_raw["year"] == replay_year]["race_name"].unique()))
        replay_race = st.selectbox("Grand Prix Race", hist_races, key="rep_race")
        
        if st.button("Run Prediction & Replay"):
                       
            df_hist = df_raw[(df_raw["year"] == replay_year) & (df_raw["race_name"] == replay_race)].copy()
            
                     
            X_hist = process_features(df_hist, preprocessor)
            probs_hist = model.predict_proba(X_hist)[:, 1]
            
            df_hist["pred_prob"] = probs_hist
            df_hist = df_hist.sort_values("pred_prob", ascending=False).reset_index(drop=True)
            df_hist["pred_rank"] = df_hist.index + 1
            
                                                                               
            df_actual = df_raw[(df_raw["year"] == replay_year) & (df_raw["race_name"] == replay_race)].copy()
            
                                                                                                               
            df_actual = df_actual.sort_values(by=["Top10", "grid"], ascending=[False, True]).reset_index(drop=True)
            df_actual["actual_rank"] = df_actual.index + 1
            
                                                                       
            actual_map = df_actual.set_index("driverRef")["Top10"].to_dict()
            df_hist["actual_Top10"] = df_hist["driverRef"].map(actual_map)
            
            col_rep1, col_rep2 = st.columns(2)
            
            with col_rep1:
                st.markdown("#### Model Predictions (Sorted by Prob)")
                st.dataframe(
                    df_hist[["pred_rank", "driverRef", "grid", "pred_prob"]]
                    .rename(columns={"pred_rank": "Rank", "driverRef": "Driver", "grid": "Start P.", "pred_prob": "Prob"})
                    .style.format({"Prob": "{:.1%}"})
                )
                
            with col_rep2:
                st.markdown("#### Actual Top 10 Finishers")
                df_act_top10 = df_actual[df_actual["Top10"] == 1][["grid", "driverRef", "constructor_name"]].reset_index(drop=True)
                df_act_top10.index = df_act_top10.index + 1
                st.dataframe(df_act_top10.rename(columns={"grid": "Start P.", "driverRef": "Driver", "constructor_name": "Team"}))
                
                                         
            predicted_top10 = set(df_hist.head(10)["driverRef"])
            actual_top10 = set(df_actual[df_actual["Top10"] == 1]["driverRef"])
            correct_count = len(predicted_top10.intersection(actual_top10))
            
            st.markdown("#### Accuracy Diagnostics")
            st.info(f"🎯 **Model accuracy for this GP**: Correctly predicted **{correct_count} out of 10** points finishers.")

                                                                                
    with tab4:
        st.subheader("🏆 Season Standing Projections")
        st.write("Simulates every round of the selected F1 season to project final Championship standings.")
        
        sim_season_year = st.selectbox("Select Season to Simulate", [2024, 2023, 2022, 2021, 2020, 2019], index=0)
        
        if st.button("Run Season Simulation"):
            df_season = df_raw[df_raw["year"] == sim_season_year].copy()
            
            if df_season.empty:
                st.error("No entries found for selected season.")
            else:
                rounds = sorted(df_season["round"].unique())
                
                              
                driver_points = {}
                constructor_points = {}
                
                                  
                points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
                
                progress_bar = st.progress(0.0)
                for i, r in enumerate(rounds):
                    df_round = df_season[df_season["round"] == r].copy()
                    
                                        
                    X_round = process_features(df_round, preprocessor)
                    probs_round = model.predict_proba(X_round)[:, 1]
                    
                    df_round["pred_prob"] = probs_round
                    df_round = df_round.sort_values("pred_prob", ascending=False).reset_index(drop=True)
                    
                                  
                    for rank_idx, row in df_round.iterrows():
                        finish_rank = rank_idx + 1
                        pts = points_map.get(finish_rank, 0)
                        
                        d_ref = row["driverRef"]
                        c_ref = row["constructor_name"]
                        
                        driver_points[d_ref] = driver_points.get(d_ref, 0) + pts
                        constructor_points[c_ref] = constructor_points.get(c_ref, 0) + pts
                        
                    progress_bar.progress((i + 1) / len(rounds))
                
                                
                sorted_drivers = sorted(driver_points.items(), key=lambda x: x[1], reverse=True)
                sorted_teams = sorted(constructor_points.items(), key=lambda x: x[1], reverse=True)
                
                col_stand1, col_stand2 = st.columns(2)
                
                with col_stand1:
                    st.markdown("#### Projected Driver Standings")
                    df_d_stand = pd.DataFrame(sorted_drivers, columns=["Driver", "Projected Points"])
                    df_d_stand.index = df_d_stand.index + 1
                    df_d_stand["Driver"] = df_d_stand["Driver"].apply(lambda x: x.capitalize())
                    st.dataframe(df_d_stand)
                    
                with col_stand2:
                    st.markdown("#### Projected Constructor Standings")
                    df_c_stand = pd.DataFrame(sorted_teams, columns=["Team", "Projected Points"])
                    df_c_stand.index = df_c_stand.index + 1
                    st.dataframe(df_c_stand)
                    
                st.success(f"🏆 Projected Champion for {sim_season_year}: **{sorted_drivers[0][0].capitalize()}**!")


if __name__ == "__main__":
    main()
