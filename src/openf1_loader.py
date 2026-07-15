"""
openf1_loader.py
================
Fetches weather data from the OpenF1 REST API (2023+).
Falls back to FastF1 weather data if unavailable.
"""
import requests
import pandas as pd
import numpy as np


_BASE_URL = "https://api.openf1.org/v1"


def _get(endpoint: str, params: dict) -> list:
    """Internal GET wrapper with error handling."""
    try:
        resp = requests.get(f"{_BASE_URL}/{endpoint}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[OpenF1Loader] Request failed for {endpoint}: {exc}")
        return []


def fetch_session_key(year: int, gp_name: str, session_type: str = "Practice 3") -> int | None:
    """
    Looks up the OpenF1 session key for a given GP and session type.

    session_type examples: 'Practice 1', 'Practice 2', 'Practice 3', 'Qualifying'
    """
    data = _get("sessions", {
        "year": year,
        "session_name": session_type,
        "meeting_name": gp_name,
    })
    if data:
        return data[0].get("session_key")
    return None


def fetch_openf1_weather(year: int, gp_name: str, session_type: str = "Practice 3") -> dict:
    """
    Fetches mean weather conditions from OpenF1 for the given session.
    Only available for 2023+.

    Returns
    -------
    dict with keys: air_temp_c, track_temp_c, humidity_pct, rainfall, wind_speed_ms
    Returns empty dict if unavailable.
    """
    session_key = fetch_session_key(year, gp_name, session_type)
    if session_key is None:
        print(f"[OpenF1Loader] No session key found for {year} {gp_name} {session_type}")
        return {}

    data = _get("weather", {"session_key": session_key})
    if not data:
        return {}

    df = pd.DataFrame(data)
    required = {"air_temperature", "track_temperature", "humidity", "rainfall"}
    if not required.issubset(df.columns):
        return {}

    return {
        "air_temp_c":    df["air_temperature"].mean(),
        "track_temp_c":  df["track_temperature"].mean(),
        "humidity_pct":  df["humidity"].mean(),
        "rainfall":      int(df["rainfall"].any()),
        "wind_speed_ms": df["wind_speed"].mean() if "wind_speed" in df.columns else np.nan,
    }
