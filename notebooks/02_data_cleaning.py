import sys
import os

                  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader import load_raw_data, merge_datasets
from data_cleaner import clean_data

def run_cleaning_pipeline():
    print("Starting F1 data cleaning pipeline...")
    
                      
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    dfs = load_raw_data(data_dir=raw_dir)
    print(f"Loaded {len(dfs)} raw datasets successfully.")
    
                       
    merged_df = merge_datasets(dfs)
    print(f"Merged master dataframe shape: {merged_df.shape}")
    
                       
    cleaned_df = clean_data(merged_df)
    print(f"Cleaned dataframe shape: {cleaned_df.shape}")
    print("\nColumns remaining after cleaning:")
    print(list(cleaned_df.columns))
    
                            
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "model_ready.csv")
    cleaned_df.to_csv(out_path, index=False)
    print(f"\nSaved cleaned dataset to: {out_path}")
    
                           
    print("\nTarget Class Distribution (Top10):")
    print(cleaned_df["Top10"].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    run_cleaning_pipeline()
