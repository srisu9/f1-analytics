import sys
import os
import pandas as pd

                  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader import load_raw_data, merge_datasets
from feature_engineer import engineer_features

def run_feature_engineering():
    print("Starting F1 feature engineering pipeline...")
    
                      
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    dfs = load_raw_data(data_dir=raw_dir)
    print(f"Loaded {len(dfs)} raw datasets successfully.")
    
                       
    merged_df = merge_datasets(dfs)
    print(f"Merged master dataframe shape: {merged_df.shape}")
    
                          
    feat_df = engineer_features(merged_df)
    print(f"Engineered dataframe shape: {feat_df.shape}")
    
    print("\nColumns in final model-ready dataset:")
    print(list(feat_df.columns))
    
                            
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "model_ready.csv")
    feat_df.to_csv(out_path, index=False)
    print(f"\nSaved feature-engineered dataset to: {out_path}")
    
                                            
    print("\nMissing values in final features:")
    print(feat_df.isnull().sum())

if __name__ == "__main__":
    run_feature_engineering()
