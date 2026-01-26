import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os

# CONFIGURATION
INPUT_PATH = 'data/processed_accidents.csv'
SAMPLE_SIZE = 50000 

def plot_feature_importance():
    print("--- STARTING FEATURE IMPORTANCE ANALYSIS ---")
    if not os.path.exists(INPUT_PATH):
        print("Error: Data file not found.")
        return

    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH)

    # 2. Clean Column Names (The Fix!)
    # This removes hidden spaces (e.g. " ID" becomes "ID")
    df.columns = df.columns.str.strip()

    # 3. Sample for speed
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

    # 4. Define Columns to DROP (The Cheating Columns)
    # We want to remove identifiers and raw timestamps
    cols_to_drop = [
        'Severity', 'ID', 'Description', 'Start_Time', 'End_Time', 
        'Weather_Timestamp', 'Airport_Code', 'Street', 'Zipcode', 
        'County', 'City', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'
    ]
    
    # Identify which of these actually exist in your dataframe
    existing_drop_cols = [col for col in cols_to_drop if col in df.columns]
    
    print(f"\n⚠️  DROPPING THESE COLUMNS: {existing_drop_cols}")
    
    # 5. Create X and y
    X = df.drop(existing_drop_cols, axis=1)
    y = df['Severity']

    # Double Check
    if 'Description' in X.columns:
        print("❌ ERROR: 'Description' column is still present!")
    else:
        print("✅ SUCCESS: 'Description' column removed.")

    # 6. Train Model
    print(f"\nTraining model on {X.shape[1]} features...")
    model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    model.fit(X, y)

    # 7. Get Feature Importance
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    
    # Sort and take Top 10
    feature_df = feature_df.sort_values(by='Importance', ascending=False).head(10)

    # 8. Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
    plt.title('Top 10 REAL Factors Influencing Accident Severity')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    
    # Save output
    os.makedirs('results', exist_ok=True)
    save_path = 'results/real_feature_importance.png'
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_feature_importance()