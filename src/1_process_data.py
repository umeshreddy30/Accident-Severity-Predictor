# FILE: src/1_process_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# CONFIGURATION
RAW_DATA_PATH = 'data/raw_accidents.csv'
PROCESSED_DATA_PATH = 'data/processed_accidents.csv'

def load_or_create_data():
    if os.path.exists(RAW_DATA_PATH):
        print(f"Loading real data from {RAW_DATA_PATH}...")
        df = pd.read_csv(RAW_DATA_PATH)
    else:
        print("Raw data not found. Generating MOCK data...")
        np.random.seed(42)
        rows = 2000
        df = pd.DataFrame({
            'Severity': np.random.choice([1, 2, 3, 4], size=rows, p=[0.1, 0.7, 0.15, 0.05]),
            'Temperature(F)': np.random.normal(60, 15, size=rows),
            'Visibility(mi)': np.random.choice([10, 5, 2, 0.5, 0.1], size=rows),
            'Weather_Condition': np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], size=rows),
            'Wind_Speed(mph)': np.random.uniform(0, 30, size=rows),
            'Day_of_Week': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], size=rows)
        })
    return df

def clean_and_encode(df):
    df = df.dropna()
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

if __name__ == "__main__":
    # THIS PART RUNS THE SCRIPT
    df = load_or_create_data()
    df_clean = clean_and_encode(df)
    os.makedirs('data', exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"SUCCESS! Processed data saved to: {PROCESSED_DATA_PATH}")
    print(df_clean.head())