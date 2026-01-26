import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# =========================================================
# CONFIGURATION - READ CAREFULLY
# This must point to your ORIGINAL 3GB file from Kaggle
# If your file is named "US_Accidents_March23.csv", change it below!
RAW_INPUT_PATH = 'data/raw_accidents.csv' 
NLP_MODEL_PATH = 'models/nlp_model.pkl'
# =========================================================

def train_nlp_from_raw():
    print(f"--- STARTING NLP RESCUE MISSION ---")
    print(f"Targeting Raw File: {RAW_INPUT_PATH}")
    
    if not os.path.exists(RAW_INPUT_PATH):
        print(f"\n❌ ERROR: Could not find '{RAW_INPUT_PATH}'")
        print("1. Check your 'data' folder.")
        print("2. Find the name of your large original CSV file.")
        print("3. Update line 12 of this script with the correct name.")
        return

    print("Reading file in chunks to extract REAL text...")
    
    collected_minor = []
    collected_major = []
    
    # We need 20k rows of each class
    target_per_class = 20000
    chunk_size = 100000
    
    try:
        # We use a chunk reader so your RAM doesn't crash
        with pd.read_csv(RAW_INPUT_PATH, chunksize=chunk_size, usecols=['Severity', 'Description']) as reader:
            for i, chunk in enumerate(reader):
                # 1. Force Text Format
                chunk['Description'] = chunk['Description'].astype(str)
                
                # 2. Filter out garbage (numbers, short text)
                # This line ensures we ONLY keep rows with actual words
                chunk = chunk[chunk['Description'].str.len() > 15] 
                chunk = chunk[~chunk['Description'].str.isnumeric()] # Remove "221608" style rows
                
                # 3. Create Target (Minor vs Major)
                chunk['Target'] = chunk['Severity'].apply(lambda x: 1 if x >= 3 else 0)
                
                # 4. Collect Data
                minor = chunk[chunk['Target'] == 0]
                major = chunk[chunk['Target'] == 1]
                
                if len(collected_minor) < target_per_class:
                    collected_minor.append(minor)
                if len(collected_major) < target_per_class:
                    collected_major.append(major)
                
                # Status Update
                curr_min = sum(len(c) for c in collected_minor)
                curr_maj = sum(len(c) for c in collected_major)
                print(f"   Chunk {i+1}: Found {curr_min} Minor / {curr_maj} Major")
                
                if curr_min >= target_per_class and curr_maj >= target_per_class:
                    print("✅ Found enough data! Stopping read.")
                    break
                    
    except ValueError as e:
        print(f"\n❌ CSV ERROR: {e}")
        print("Are you sure the columns are named 'Severity' and 'Description' in this file?")
        return

    # Assemble
    print("Combining data...")
    df_minor = pd.concat(collected_minor).sample(n=target_per_class, random_state=42)
    df_major = pd.concat(collected_major).sample(n=target_per_class, random_state=42)
    
    df_balanced = pd.concat([df_minor, df_major])
    
    print("\n--- DATA QUALITY CHECK ---")
    print("These should be WORDS, not numbers:")
    print(df_balanced['Description'].head(3).tolist())
    
    X = df_balanced['Description']
    y = df_balanced['Target']

    # Train
    print("\nTraining XGBoost on REAL text...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,3))),
        ('clf', XGBClassifier(n_estimators=100, max_depth=6, eval_metric='logloss'))
    ])
    
    pipeline.fit(X_train, y_train)

    # Evaluate
    print("\n--- FINAL REPORT ---")
    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds, target_names=['Minor', 'Major']))
    
    # Save
    joblib.dump(pipeline, NLP_MODEL_PATH)
    print(f"✅ Saved fixed model to {NLP_MODEL_PATH}")

    # Test
    print("\n--- SANITY CHECK (FINAL) ---")
    test_phrases = [
        "Slow traffic due to construction work.",
        "Severe collision, vehicle overturned, heavy fire.",
        "Incident on highway, right lane blocked."
    ]
    for phrase in test_phrases:
        pred = pipeline.predict([phrase])[0]
        label = "🚨 MAJOR" if pred == 1 else "✅ MINOR"
        print(f"'{phrase}' -> {label}")

if __name__ == "__main__":
    train_nlp_from_raw()