import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# =============================================
# CONFIGURATION
RAW_INPUT_PATH = 'data/raw_accidents.csv' 
NLP_MODEL_PATH = 'models/nlp_model.pkl'
# =============================================

def train_smart_nlp():
    print("--- STARTING SMART NLP TRAINING (KEYWORD GUIDED) ---")
    
    if not os.path.exists(RAW_INPUT_PATH):
        print(f"❌ Error: '{RAW_INPUT_PATH}' not found.")
        return

    # 1. READ RAW DATA (Load more rows to find specific keywords)
    print(f"Reading raw data from {RAW_INPUT_PATH}...")
    try:
        # We read 300,000 rows to ensure we find enough "perfect" examples
        df = pd.read_csv(RAW_INPUT_PATH, nrows=300000, usecols=['Severity', 'Description'])
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # 2. CLEANING
    print("Cleaning text...")
    df['Description'] = df['Description'].astype(str).str.lower()
    df = df[df['Description'].str.len() > 15]
    df = df[~df['Description'].str.isnumeric()]
    
    # 3. SMART SAMPLING (The Fix)
    print("Filtering for high-quality examples...")
    
    # Define words that STRONGLY indicate each class
    major_keywords = ['closed', 'blocked', 'fire', 'overturned', 'injury', 'severe', 'medical', 'fatal', 'lifeflight']
    minor_keywords = ['slow', 'delay', 'shoulder', 'maintenance', 'construction', 'stopped', 'queuing', 'moderate']
    
    # Create regex patterns
    major_pattern = '|'.join(major_keywords)
    minor_pattern = '|'.join(minor_keywords)
    
    # Filter: Keep Major only if it has a Major keyword, Minor only if it has a Minor keyword
    # This removes confusing/generic descriptions like "Accident on I-95."
    df_major_clean = df[
        (df['Severity'] >= 3) & 
        (df['Description'].str.contains(major_pattern))
    ]
    
    df_minor_clean = df[
        (df['Severity'] < 3) & 
        (df['Description'].str.contains(minor_pattern))
    ]
    
    # 4. BALANCE
    print("Balancing classes...")
    # Take up to 25,000 of each
    limit = min(len(df_major_clean), len(df_minor_clean), 25000)
    
    print(f"Found {len(df_major_clean)} good Major examples and {len(df_minor_clean)} good Minor examples.")
    
    df_bal = pd.concat([
        df_minor_clean.sample(limit, random_state=42),
        df_major_clean.sample(limit, random_state=42)
    ])
    
    # Create Target (0=Minor, 1=Major)
    df_bal['Target'] = df_bal['Severity'].apply(lambda x: 1 if x >= 3 else 0)
    
    print(f"Training on {len(df_bal)} balanced, high-quality rows.")

    # 5. TRAIN PIPELINE
    print("Training Model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))),
        # Increased 'C' to 2.0 to make the model more decisive
        ('clf', LogisticRegression(max_iter=1000, C=2.0, solver='liblinear'))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(df_bal['Description'], df_bal['Target'], test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # 6. EVALUATE
    print("\n--- RESULTS ---")
    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds, target_names=['Minor', 'Major']))
    
    # 7. SAVE
    joblib.dump(pipeline, NLP_MODEL_PATH)
    print(f"✅ Smart Model saved to {NLP_MODEL_PATH}")

    # 8. TEST
    print("\n--- SANITY CHECK ---")
    test_phrases = [
        "Slow traffic due to construction work on shoulder.",
        "Severe collision, vehicle overturned, heavy fire."
    ]
    for phrase in test_phrases:
        prob = pipeline.predict_proba([phrase])[0][1] * 100
        label = "🚨 MAJOR" if prob > 50 else "✅ MINOR"
        print(f"'{phrase}' -> {label} (Risk: {prob:.1f}%)")

if __name__ == "__main__":
    train_smart_nlp()