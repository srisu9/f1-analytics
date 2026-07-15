"""
fastf1_loader.py
================
Wrapper around the FastF1 library for caching and extracting:
  - Session lap times (FP1, FP2, FP3, Q, R)
  - Telemetry aggregates (speed, throttle, brake)
  - Weather recorded during sessions

All data returned is pre-race only (practice + qualifying).
FastF1 covers 2018 onwards reliably.
"""
import os
import pandas as pd
import numpy as np

# FastF1 cache path — put outside OneDrive to avoid sync conflicts
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fastf1")
_CACHE_DIR = os.path.abspath(_CACHE_DIR)

_ff1_available = False
try:
    import fastf1
    fastf1.Cache.enable_cache(_CACHE_DIR)
    _ff1_available = True
    print(f"[FastF1Loader] Cache enabled at: {_CACHE_DIR}")
except ImportError:
    print("[FastF1Loader] WARNING: fastf1 not installed. Functions will return empty DataFrames.")


def _load_session(year: int, gp: str, session_type: str):
    """Internal helper to load and cache a FastF1 session."""
    if not _ff1_available:
        return None
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(laps=True, telemetry=False, weather=True, messages=False)
        return session
    except Exception as exc:
        print(f"[FastF1Loader] Could not load {year} {gp} {session_type}: {exc}")
        return None


def fetch_session_best_laps(year: int, gp: str, session_type: str = "FP1") -> pd.DataFrame:
    """
    Returns the best (fastest) lap time per driver for a given session.

    Returns
    -------
    DataFrame with columns:
        driverRef, session_best_laptime_s (seconds), session_type, year, gp
    """
    session = _load_session(year, gp, session_type)
    if session is None:
        return pd.DataFrame()

    try:
        laps = session.laps
        # Filter to personal best laps only, remove outliers
        laps = laps.pick_quicklaps()
        best = (
            laps.groupby("Driver")["LapTime"]
            .min()
            .reset_index()
            .rename(columns={"Driver": "driver_code"})
        )
        best["session_best_laptime_s"] = best["LapTime"].dt.total_seconds()
        best["session_type"] = session_type
        best["year"] = year
        best["gp"] = gp
        return best[["driver_code", "session_best_laptime_s", "session_type", "year", "gp"]]
    except Exception as exc:
        print(f"[FastF1Loader] Error processing laps for {year} {gp} {session_type}: {exc}")
        return pd.DataFrame()


def fetch_fp2_longrun_pace(year: int, gp: str, min_consecutive: int = 5) -> pd.DataFrame:
    """
    Estimates long-run race pace from FP2 by finding each driver's
    longest consecutive stint and computing mean lap time.

    A 'long run' is defined as >= min_consecutive clean laps on the same tyre.

    Returns
    -------
    DataFrame with columns:
        driver_code, fp2_longrun_avg_s, fp2_longrun_laps, year, gp
    """
    session = _load_session(year, gp, "FP2")
    if session is None:
        return pd.DataFrame()

    try:
        laps = session.laps.pick_quicklaps().copy()
        laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

        results = []
        for driver, grp in laps.groupby("Driver"):
            grp = grp.sort_values("LapNumber").reset_index(drop=True)
            # Find longest stint on same compound
            grp["stint"] = (grp["Compound"] != grp["Compound"].shift()).cumsum()
            for stint_id, stint_df in grp.groupby("stint"):
                if len(stint_df) >= min_consecutive:
                    results.append({
                        "driver_code": driver,
                        "fp2_longrun_avg_s": stint_df["LapTime_s"].mean(),
                        "fp2_longrun_laps": len(stint_df),
                        "year": year,
                        "gp": gp
                    })

        if not results:
            return pd.DataFrame()

        # Keep only the longest qualifying stint per driver
        df = pd.DataFrame(results)
        df = df.sort_values("fp2_longrun_laps", ascending=False)
        df = df.drop_duplicates(subset="driver_code", keep="first")
        return df[["driver_code", "fp2_longrun_avg_s", "fp2_longrun_laps", "year", "gp"]]

    except Exception as exc:
        print(f"[FastF1Loader] FP2 long-run error {year} {gp}: {exc}")
        return pd.DataFrame()


def fetch_qualifying_sectors(year: int, gp: str) -> pd.DataFrame:
    """
    Returns fastest sector times per driver from qualifying (best across Q1/Q2/Q3).

    Returns
    -------
    DataFrame with columns:
        driver_code, quali_sector1_s, quali_sector2_s, quali_sector3_s,
        quali_pole_delta_s, year, gp
    """
    session = _load_session(year, gp, "Q")
    if session is None:
        return pd.DataFrame()

    try:
        laps = session.laps.pick_quicklaps()
        laps = laps.copy()
        for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
            laps[col + "_s"] = laps[col].dt.total_seconds()
        laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

        best = laps.groupby("Driver").agg(
            quali_sector1_s=("Sector1Time_s", "min"),
            quali_sector2_s=("Sector2Time_s", "min"),
            quali_sector3_s=("Sector3Time_s", "min"),
            quali_best_lap_s=("LapTime_s", "min"),
        ).reset_index().rename(columns={"Driver": "driver_code"})

        pole_time = best["quali_best_lap_s"].min()
        best["quali_pole_delta_s"] = best["quali_best_lap_s"] - pole_time
        best["year"] = year
        best["gp"] = gp

        return best[[
            "driver_code", "quali_sector1_s", "quali_sector2_s", "quali_sector3_s",
            "quali_pole_delta_s", "year", "gp"
        ]]

    except Exception as exc:
        print(f"[FastF1Loader] Qualifying sectors error {year} {gp}: {exc}")
        return pd.DataFrame()


def fetch_telemetry_aggregates(year: int, gp: str, session_type: str = "FP2") -> pd.DataFrame:
    """
    Aggregates car telemetry from a practice session into scalar features per driver.

    Loads telemetry separately (heavier call). Aggregates:
      - avg_speed_kph: mean speed across all sampled points
      - top_speed_kph: max speed recorded
      - throttle_pct: % of samples with throttle >= 98%
      - brake_pct: % of samples with brake == True

    Returns
    -------
    DataFrame with columns:
        driver_code, avg_speed_kph, top_speed_kph, throttle_pct, brake_pct, year, gp
    """
    if not _ff1_available:
        return pd.DataFrame()

    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(laps=True, telemetry=True, weather=False, messages=False)

        laps = session.laps.pick_quicklaps()
        results = []

        for driver in laps["Driver"].unique():
            try:
                driver_laps = laps.pick_driver(driver)
                # Sample first 3 fastest laps to keep computation light
                driver_laps = driver_laps.sort_values("LapTime").head(3)
                tel_frames = []
                for _, lap in driver_laps.iterlaps():
                    tel = lap.get_telemetry()
                    if tel is not None and len(tel) > 0:
                        tel_frames.append(tel)

                if not tel_frames:
                    continue

                tel_all = pd.concat(tel_frames, ignore_index=True)

                results.append({
                    "driver_code": driver,
                    "avg_speed_kph": tel_all["Speed"].mean(),
                    "top_speed_kph": tel_all["Speed"].max(),
                    "throttle_pct": (tel_all["Throttle"] >= 98).mean() * 100,
                    "brake_pct": tel_all["Brake"].mean() * 100 if "Brake" in tel_all.columns else np.nan,
                    "year": year,
                    "gp": gp,
                })
            except Exception:
                continue

        return pd.DataFrame(results) if results else pd.DataFrame()

    except Exception as exc:
        print(f"[FastF1Loader] Telemetry error {year} {gp} {session_type}: {exc}")
        return pd.DataFrame()


def fetch_sector_consistency(year: int, gp: str, session_type: str = "Q") -> pd.DataFrame:
    """
    Measures lap-to-lap sector time consistency (std dev) from qualifying.
    Lower std = more consistent driver.

    Returns
    -------
    DataFrame with columns:
        driver_code, sector1_std_s, sector2_std_s, sector3_std_s, year, gp
    """
    session = _load_session(year, gp, session_type)
    if session is None:
        return pd.DataFrame()

    try:
        laps = session.laps.pick_quicklaps().copy()
        for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
            laps[col + "_s"] = laps[col].dt.total_seconds()

        std_df = laps.groupby("Driver").agg(
            sector1_std_s=("Sector1Time_s", "std"),
            sector2_std_s=("Sector2Time_s", "std"),
            sector3_std_s=("Sector3Time_s", "std"),
        ).reset_index().rename(columns={"Driver": "driver_code"})

        std_df["year"] = year
        std_df["gp"] = gp
        return std_df

    except Exception as exc:
        print(f"[FastF1Loader] Sector consistency error {year} {gp}: {exc}")
        return pd.DataFrame()


def fetch_session_weather(year: int, gp: str, session_type: str = "FP3") -> dict:
    """
    Returns mean weather conditions recorded during a session.
    FP3 is the closest pre-race session and best proxy for race conditions.

    Returns
    -------
    dict with keys:
        air_temp_c, track_temp_c, humidity_pct, rainfall (bool), wind_speed_ms
    Returns empty dict on failure.
    """
    session = _load_session(year, gp, session_type)
    if session is None:
        return {}

    try:
        weather = session.weather_data
        if weather is None or weather.empty:
            return {}

        return {
            "air_temp_c":    weather["AirTemp"].mean(),
            "track_temp_c":  weather["TrackTemp"].mean(),
            "humidity_pct":  weather["Humidity"].mean(),
            "rainfall":      int(weather["Rainfall"].any()),
            "wind_speed_ms": weather["WindSpeed"].mean() if "WindSpeed" in weather.columns else np.nan,
        }
    except Exception as exc:
        print(f"[FastF1Loader] Weather error {year} {gp} {session_type}: {exc}")
        return {}
