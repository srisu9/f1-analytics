import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the merged F1 dataframe:
    - Creates target variable Top10.
    - Drops leakage, redundant, URL, and practice columns.
    - Handles datetime conversions.
    - Cleans numerical features.
    """
    cleaned_df = df.copy()
    
                                     
    cleaned_df["Top10"] = (cleaned_df["positionOrder"] <= 10).astype(int)
    
                                         
    cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce")
    cleaned_df["dob"] = pd.to_datetime(cleaned_df["dob"], errors="coerce")
    
                              
    numeric_cols = ["grid", "qualifying_position", "car_number", "alt"]
    for col in numeric_cols:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
        
                               
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
    
                 
    id_cols = [
        "resultId", 
        "raceId", 
        "driverId", 
        "constructorId", 
        "circuitId"
    ]
    
          
    url_cols = [
        "driver_url", 
        "constructor_url", 
        "race_url", 
        "circuit_url"
    ]
    
                              
    practice_sprint_cols = [
        "fp1_date", "fp1_time", 
        "fp2_date", "fp2_time", 
        "fp3_date", "fp3_time", 
        "sprint_date", "sprint_time"
    ]
    
                                   
    redundant_cols = [
        "quali_date", 
        "quali_time", 
        "race_time", 
        "constructorRef", 
        "quali_car_number",
        "driver_number",
        "q1", "q2", "q3"
    ]
    
    drop_cols = leakage_cols + id_cols + url_cols + practice_sprint_cols + redundant_cols
    cleaned_df.drop(columns=drop_cols, inplace=True, errors="ignore")
    
    return cleaned_df
