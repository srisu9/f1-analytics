import pandas as pd
import numpy as np
import os

def load_raw_data(data_dir="data/raw"):
    """
    Loads all raw Ergast F1 CSV datasets and returns them.
    Replaces '\\N' string placeholders with np.nan.
    """
    files = {
        "drivers": "drivers.csv",
        "constructors": "constructors.csv",
        "races": "races.csv",
        "results": "results.csv",
        "qualifying": "qualifying.csv",
        "circuits": "circuits.csv"
    }
    
    dfs = {}
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing F1 dataset: {filepath}")
        
        # Load CSV and treat '\N' as NaN
        df = pd.read_csv(filepath, na_values=['\\N', '\\\\N', 'N', 'n/a', 'nan', 'NaN'])
        dfs[name] = df
        
    return dfs

def merge_datasets(dfs):
    """
    Performs data merging to create a master dataframe.
    Pre-renames columns to avoid suffix conflicts.
    """
    # 1. Prepare Drivers table
    drivers_df = dfs["drivers"].copy()
    drivers_df.rename(columns={
        "number": "driver_number",
        "url": "driver_url",
        "nationality": "driver_nationality"
    }, inplace=True)
    
    # 2. Prepare Constructors table
    constructors_df = dfs["constructors"].copy()
    constructors_df.rename(columns={
        "name": "constructor_name",
        "url": "constructor_url",
        "nationality": "constructor_nationality"
    }, inplace=True)
    
    # 3. Prepare Races table
    races_df = dfs["races"].copy()
    races_df.rename(columns={
        "name": "race_name",
        "time": "race_time",
        "url": "race_url"
    }, inplace=True)
    
    # 4. Prepare Qualifying table
    qualifying_df = dfs["qualifying"].copy()
    qualifying_df.rename(columns={
        "number": "quali_car_number",
        "position": "qualifying_position"
    }, inplace=True)
    # drop qualifyId in pre-merge to avoid conflict
    qualifying_df.drop(columns=["qualifyId"], inplace=True, errors="ignore")
    
    # 5. Prepare Circuits table
    circuits_df = dfs["circuits"].copy()
    circuits_df.rename(columns={
        "name": "circuit_name",
        "url": "circuit_url"
    }, inplace=True)
    
    # Start with results table as base
    df = dfs["results"].copy()
    # rename results number to car_number to be clear
    df.rename(columns={"number": "car_number"}, inplace=True)
    
    # Merge sequentially
    df = pd.merge(df, drivers_df, on="driverId", how="left")
    df = pd.merge(df, constructors_df, on="constructorId", how="left")
    df = pd.merge(df, races_df, on="raceId", how="left")
    df = pd.merge(df, qualifying_df, on=["raceId", "driverId", "constructorId"], how="left")
    df = pd.merge(df, circuits_df, on="circuitId", how="left")
    
    return df
