import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import os
import time
import joblib  # Used to save the model

# CONFIGURATION
INPUT_PATH = 'data/processed_accidents.csv'
MODEL_PATH = 'models/accident_model.pkl'
SAMPLE_SIZE = 100000  # Limit to 100k rows for speed

def train():
    start_time = time.time()
    
    # 1. Load Data
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found. Run '1_process_data.py' first.")
        return

    print("Loading data...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Original dataset shape: {df.shape}")

    # 2. SAMPLING (Speed Fix)
    if len(df) > SAMPLE_SIZE:
        print(f"⚠️ Dataset is too large for local training. Sampling down to {SAMPLE_SIZE} rows...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    # 3. Split Features (X) and Target (y)
    # We must remove the same columns we ignored in visualization to be consistent, 
    # BUT for the main prediction model, we usually keep as much info as possible 
    # except for IDs and timestamps that leak answers.
    
    # Drop "Cheating" columns if they exist
    drop_cols = ['ID', 'Description', 'Start_Time', 'End_Time', 'Zipcode']
    existing_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(existing_drop, axis=1)

    X = df.drop('Severity', axis=1)
    y = df['Severity']

    # 4. Split into Train and Test sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Handle Imbalance (SMOTE)
    print("Applying SMOTE (this might take a moment)...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Training data shape after SMOTE: {X_train_smote.shape}")

    # 6. Train Model (Random Forest)
    print("Training Random Forest Model (using all CPU cores)...")
    model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    model.fit(X_train_smote, y_train_smote)

    # 7. Evaluate
    print("Evaluating model...")
    predictions = model.predict(X_test)
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, predictions))
    
    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_test, predictions))
    
    # 8. SAVE MODEL (The New Part)
    print("\nSaving model to disk...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    print(f"\nTotal execution time: {round(time.time() - start_time, 2)} seconds")

if __name__ == "__main__":
    train()