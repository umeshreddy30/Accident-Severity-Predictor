import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# CONFIGURATION
INPUT_PATH = 'data/processed_accidents.csv'
MODEL_PATH = 'models/demo_model.pkl' # <--- New filename
SAMPLE_SIZE = 100000

# THESE ARE THE ONLY COLUMNS WE WANT THE APP TO USE
APP_FEATURES = [
    'Start_Lat', 'Start_Lng', 
    'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 
    'Distance(mi)'
]

def train_demo_model():
    print("Loading data for Demo Model...")
    if not os.path.exists(INPUT_PATH):
        print("Error: Data file not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    
    # 1. Sample for speed
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

    # 2. Filter ONLY the columns we have sliders for
    # We check which ones actually exist in the CSV
    existing_features = [col for col in APP_FEATURES if col in df.columns]
    
    print(f"Training on these specific features: {existing_features}")
    
    X = df[existing_features]
    y = df['Severity']

    # 3. Handle Imbalance (Crucial for getting varied results)
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # 4. Train Model
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_res, y_res)

    # 5. Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ DEMO Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_demo_model()