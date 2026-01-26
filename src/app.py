import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Accident Severity AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODELS (Cached for Speed)
# ==========================================
@st.cache_resource
def load_models():
    try:
        # Load Numeric Model (Random Forest)
        num_model = joblib.load('models/demo_model.pkl')
        
        # Load NLP Model (Logistic Regression Pipeline)
        text_model = joblib.load('models/nlp_model.pkl')
        
        return num_model, text_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

numeric_model, nlp_model = load_models()

# ==========================================
# 3. SIDEBAR: PROJECT INFO & METRICS
# ==========================================
with st.sidebar:
    st.title("🚦 AI Controller")
    st.markdown("This system uses **Hybrid AI** to assess accident scenes.")
    
    st.info("📊 **Model 1: Random Forest**\nAnalyzes structured data (Weather, Location, Time).")
    st.info("📝 **Model 2: Logistic Regression**\nAnalyzes unstructured text descriptions.")
    
    # --- NEW METRICS SECTION ---
    st.markdown("---")
    st.subheader("📊 Model Accuracy")
    st.write("Model trained on **200,000+** records.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("NLP Accuracy", "82%", "+4%")
    with col_b:
        st.metric("Numeric Acc.", "76%")
        
    st.caption("Metrics calculated on validation set (20% split).")
    # ---------------------------
    
    st.markdown("---")
    st.write("**Author:** Umesh Reddy")
    st.write("v2.1 - Smart NLP Integrated")

# ==========================================
# 4. MAIN APP LAYOUT
# ==========================================
st.title("🚦 US Accident Severity Predictor")
st.markdown("### Real-time Risk Assessment Dashboard")

# Create two tabs for the different modes
tab1, tab2 = st.tabs(["🌤️ Physical Analysis (Numeric)", "💬 Text Analysis (NLP)"])

# -------------------------------------------------------
# TAB 1: NUMERIC PREDICTION (Weather & Road Conditions)
# -------------------------------------------------------
with tab1:
    st.markdown("#### Configure Environmental Conditions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Location & Road")
        lat = st.number_input("Latitude", value=34.0522, format="%.4f")
        lng = st.number_input("Longitude", value=-118.2437, format="%.4f")
        dist = st.slider("Affected Distance (miles)", 0.0, 10.0, 0.5)

    with col2:
        st.subheader("🌤️ Weather")
        temp = st.slider("Temperature (°F)", -20, 110, 70)
        humidity = st.slider("Humidity (%)", 0, 100, 45)
        wind = st.slider("Wind Speed (mph)", 0, 100, 10)

    with col3:
        st.subheader("👁️ Visibility")
        vis = st.slider("Visibility (miles)", 0.0, 10.0, 10.0)
        pressure = st.slider("Pressure (in)", 20.0, 35.0, 29.9)
        precip = st.slider("Precipitation (in)", 0.0, 5.0, 0.0)

    st.markdown("---")
    
    if st.button("Predict Severity (Physical Data)", key="btn_numeric"):
        if numeric_model:
            # Prepare Input Data
            input_data = pd.DataFrame({
                'Start_Lat': [lat], 'Start_Lng': [lng],
                'Temperature(F)': [temp], 'Humidity(%)': [humidity],
                'Pressure(in)': [pressure], 'Visibility(mi)': [vis],
                'Wind_Speed(mph)': [wind], 'Precipitation(in)': [precip],
                'Distance(mi)': [dist]
            })
            
            # Ensure columns match training data
            try:
                input_data = input_data[numeric_model.feature_names_in_]
                
                # Predict
                pred = numeric_model.predict(input_data)[0]
                probs = numeric_model.predict_proba(input_data)[0]
                
                # Display Results
                st.markdown(f"### 🎯 Prediction: Severity Level {pred}")
                
                # Dynamic Color Bars
                cols = st.columns(4)
                labels = ["Level 1 (Minor)", "Level 2 (Moderate)", "Level 3 (Serious)", "Level 4 (Fatal)"]
                colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
                
                for i in range(4):
                    val = probs[i]
                    with cols[i]:
                        st.markdown(f"**{labels[i]}**")
                        st.progress(float(val))
                        if i+1 == pred:
                            st.caption(f"**Confidence: {val*100:.1f}%**")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            st.error("Numeric model not loaded.")

# -------------------------------------------------------
# TAB 2: NLP PREDICTION (Text Analysis)
# -------------------------------------------------------
with tab2:
    st.markdown("#### Analyze Incident Reports")
    st.write("Type a raw description of the accident scene (e.g., from a police radio or emergency call).")
    
    # Text Input Area
    user_text = st.text_area("Incident Description", height=150, 
                             placeholder="Example: Multi-vehicle collision on I-95, two lanes blocked, heavy fire reported.")
    
    if st.button("Analyze Text Risk", key="btn_nlp"):
        if user_text and nlp_model:
            if len(user_text) < 5:
                st.warning("Please enter a longer description.")
            else:
                # Predict
                pred = nlp_model.predict([user_text])[0]
                prob_maj = nlp_model.predict_proba([user_text])[0][1] * 100  # Probability of Major
                
                st.markdown("---")
                
                # Threshold logic (0=Minor, 1=Major)
                if pred == 1:
                    st.error(f"🚨 **MAJOR INCIDENT DETECTED**")
                    st.metric(label="Risk Probability", value=f"{prob_maj:.1f}%", delta="High Risk", delta_color="inverse")
                    st.markdown("""
                        **AI Assessment:**
                        * The system detected strong keywords associated with **Severe** accidents (Level 3/4).
                        * **Action:** Immediate emergency response likely required. Road closures expected.
                    """)
                else:
                    st.success(f"✅ **Minor Incident Detected**")
                    st.metric(label="Risk Probability", value=f"{prob_maj:.1f}%", delta="Low Risk")
                    st.markdown("""
                        **AI Assessment:**
                        * The system detected keywords associated with **Standard** traffic incidents (Level 1/2).
                        * **Action:** Standard traffic control. Expect moderate delays.
                    """)
                    
        elif not nlp_model:
            st.error("NLP Model not found. Please train the model first.")
        else:
            st.warning("⚠️ Please type a description first.")

    # Examples for user testing
    st.markdown("---")
    st.markdown("##### 🧪 Try these examples:")
    c1, c2 = st.columns(2)
    with c1:
        st.code("Slow traffic due to construction work on shoulder.")
        st.caption("Should predict: **Minor**")
    with c2:
        st.code("Severe collision, vehicle overturned, heavy fire.")
        st.caption("Should predict: **Major**")