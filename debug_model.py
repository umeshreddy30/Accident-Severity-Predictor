import joblib
import os

MODEL_PATH = 'models/nlp_model.pkl'

print(f"--- DIAGNOSING MODEL AT: {MODEL_PATH} ---")

# 1. Check if file exists
if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: Model file does not exist!")
    exit()

# 2. Load the model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load model. {e}")
    exit()

# 3. Check the internal vocabulary (Did it learn words?)
try:
    # Access the vectorizer step inside the pipeline
    vocab_size = len(model.named_steps['tfidf'].vocabulary_)
    print(f"📚 Vocabulary Size: {vocab_size} words learned.")
    
    if vocab_size < 100:
        print("⚠️ WARNING: Vocabulary is extremely small. Training might have failed.")
except Exception as e:
    print(f"⚠️ Could not check vocabulary: {e}")

# 4. Test Predictions
print("\n--- LIVE TEST ---")
test_inputs = [
    "Severe major accident with fire and overturned vehicle.",  # Should be HIGH
    "Minor delay due to road work.",                          # Should be LOW
    "Two lanes blocked.",                                     # Should be HIGH/MEDIUM
    "Clear conditions."                                       # Should be LOW
]

for text in test_inputs:
    try:
        prob = model.predict_proba([text])[0][1] * 100
        pred = model.predict([text])[0]
        result = "🚨 MAJOR" if pred == 1 else "✅ MINOR"
        print(f"Input: '{text}'")
        print(f"   -> Prediction: {result} (Risk: {prob:.2f}%)")
    except Exception as e:
        print(f"   -> Error predicting: {e}")

print("\n------------------------------------------------")
print("VERDICT:")
print("If the risks above are all ~50%, the training failed.")
print("If they vary (e.g., 20% vs 90%), the model is fine (Clear Streamlit Cache!).")