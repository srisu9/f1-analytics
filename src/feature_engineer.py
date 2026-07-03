import pandas as pd
import numpy as np

def engineer_features(merged_df):
    """
    Computes features using only pre-race information.
    Takes the RAW merged dataframe (which has positionOrder and points)
    so it can calculate rolling averages and standings, and then drops
    leakage/helper columns at the end.
    """
    df = merged_df.copy()
    
    # Ensure chronological order for rolling/cumulative features
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(by=['date', 'round']).reset_index(drop=True)
    
    # 1. Target Variable
    df["Top10"] = (df["positionOrder"] <= 10).astype(int)
    
    # 2. Driver Age
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25
    
    # 3. Driver Experience (cumulative races started)
    # cumcount starts at 0, which represents "0 prior races" (perfect for pre-race feature!)
    df['driver_experience'] = df.groupby('driverRef').cumcount()
    
    # 4. Driver Win Rate (prior Top 10 finish rate)
    df['driver_prev_top10_sum'] = df.groupby('driverRef')['Top10'].transform(lambda x: x.cumsum().shift(1).fillna(0))
    df['driver_prev_races_count'] = df.groupby('driverRef').cumcount()
    df['driver_win_rate'] = df['driver_prev_top10_sum'] / df['driver_prev_races_count'].replace(0, np.nan)
    # Fill first-race drivers with average F1 Top 10 rate (~0.42)
    df['driver_win_rate'] = df['driver_win_rate'].fillna(0.42)
    
    # 5. Constructor Win Rate (prior Top 10 finish rate)
    df['constructor_prev_top10_sum'] = df.groupby('constructor_name')['Top10'].transform(lambda x: x.cumsum().shift(1).fillna(0))
    df['constructor_prev_races_count'] = df.groupby('constructor_name').cumcount()
    df['constructor_win_rate'] = df['constructor_prev_top10_sum'] / df['constructor_prev_races_count'].replace(0, np.nan)
    df['constructor_win_rate'] = df['constructor_win_rate'].fillna(0.42)
    
    # Clean types for positionOrder
    df['positionOrder'] = pd.to_numeric(df['positionOrder'], errors='coerce')
    
    # 6. Driver Rolling average finish positions (prior 3 and prior 5 races)
    # shift(1) is critical to exclude the current race result
    df['rolling_avg_finish_3'] = df.groupby('driverRef')['positionOrder'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    )
    df['rolling_avg_finish_5'] = df.groupby('driverRef')['positionOrder'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    # For first-race or early-career drivers, fill with standard midpoint finish (e.g. 12th)
    df['rolling_avg_finish_3'] = df['rolling_avg_finish_3'].fillna(12.0)
    df['rolling_avg_finish_5'] = df['rolling_avg_finish_5'].fillna(12.0)
    
    # 7. Previous Race Finish position
    df['prev_race_finish'] = df.groupby('driverRef')['positionOrder'].shift(1).fillna(12.0)
    
    # 8. Home Race (1 if driver nationality matches circuit country)
    nationality_to_country = {
        "British": "UK",
        "German": "Germany",
        "French": "France",
        "Italian": "Italy",
        "Spanish": "Spain",
        "Australian": "Australia",
        "Austrian": "Austria",
        "Japanese": "Japan",
        "Brazilian": "Brazil",
        "Canadian": "Canada",
        "American": "USA",
        "Belgian": "Belgium",
        "Dutch": "Netherlands",
        "Monaco": "Monaco",
        "Mexican": "Mexico",
        "Finnish": "Finland",
        "Swiss": "Switzerland",
        "Russian": "Russia",
        "Spanish": "Spain",
        "Swedish": "Sweden",
        "New Zealander": "New Zealand",
        "South African": "South Africa"
    }
    driver_country_mapped = df['driver_nationality'].map(nationality_to_country)
    df['home_race'] = (driver_country_mapped == df['country']).astype(int)
    
    # 9. Grid vs Qualifying position difference
    df['qualifying_position'] = pd.to_numeric(df['qualifying_position'], errors='coerce')
    df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
    
    # If qualifying_position is missing, fill it with starting grid (since grid position is a good proxy)
    df['qualifying_position_imputed'] = df['qualifying_position'].fillna(df['grid'])
    df['grid_qualifying_diff'] = df['grid'] - df['qualifying_position_imputed']
    
    # 10. Constructor Season Points (cumulative season points up to previous round)
    # First get points by constructor per round
    df['points'] = pd.to_numeric(df['points'], errors='coerce').fillna(0.0)
    round_points = df.groupby(['year', 'round', 'constructor_name'])['points'].sum().reset_index()
    # Sort chronologically to cumulative sum
    round_points = round_points.sort_values(by=['year', 'round'])
    # Cum sum grouped by year & constructor, then shift to get points UP TO the previous round
    round_points['constructor_season_points'] = round_points.groupby(['year', 'constructor_name'])['points'].transform(
        lambda x: x.cumsum().shift(1).fillna(0.0)
    )
    # Drop points column to avoid merge conflict
    round_points.drop(columns=['points'], inplace=True)
    # Merge back
    df = pd.merge(df, round_points, on=['year', 'round', 'constructor_name'], how='left')
    df['constructor_season_points'] = df['constructor_season_points'].fillna(0.0)
    
    # Drop helper/intermediate columns used for calculation
    drop_helpers = [
        'driver_prev_top10_sum', 'driver_prev_races_count',
        'constructor_prev_top10_sum', 'constructor_prev_races_count',
        'qualifying_position_imputed'
    ]
    df.drop(columns=drop_helpers, inplace=True, errors='ignore')
    
    # Drop post-race outcome/leakage columns
    leakage_cols = [
        "position", 
        "positionOrder", 
        "positionText", 
        "points", 
        "time", 
        "milliseconds", 
        "fastestLap", 
        "rank", 
        "fastestLapTime", 
        "fastestLapSpeed",
        "laps",
        "statusId"
    ]
    df.drop(columns=leakage_cols, inplace=True, errors="ignore")
    
    # Drop ID columns
    id_cols = ["resultId", "raceId", "driverId", "constructorId", "circuitId"]
    df.drop(columns=id_cols, inplace=True, errors="ignore")
    
    # Drop raw details we don't need for ML or have already engineered
    details_cols = [
        "fp1_date", "fp1_time", "fp2_date", "fp2_time", "fp3_date", "fp3_time",
        "sprint_date", "sprint_time", "quali_date", "quali_time", "race_time",
        "driver_url", "constructor_url", "race_url", "circuit_url",
        "constructorRef", "quali_car_number", "driver_number", "q1", "q2", "q3",
        "code", "forename", "surname", "dob"
    ]
    df.drop(columns=details_cols, inplace=True, errors="ignore")
    
    return df
