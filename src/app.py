import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the DEMO model (The one we just trained)
model = joblib.load('models/demo_model.pkl')

st.title("🚦 Traffic Accident Severity Predictor")
st.write("Adjust the sliders to predict accident severity.")

st.sidebar.header("User Input Features")

# A. Location (Default to NYC area)
st.sidebar.subheader("📍 Location")
lat = st.sidebar.number_input("Latitude", value=40.7128)
lng = st.sidebar.number_input("Longitude", value=-74.0060)

# B. Weather
st.sidebar.subheader("🌤️ Weather")
temperature = st.sidebar.slider("Temperature (F)", -20.0, 110.0, 70.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 40)
pressure = st.sidebar.slider("Pressure (in)", 20.0, 35.0, 29.0)
visibility = st.sidebar.slider("Visibility (miles)", 0.0, 10.0, 10.0)
wind_speed = st.sidebar.slider("Wind Speed (mph)", 0.0, 100.0, 10.0)
precip = st.sidebar.slider("Precipitation (in)", 0.0, 10.0, 0.0)

# C. Road
st.sidebar.subheader("🛣️ Road Config")
distance = st.sidebar.slider("Length of Road Affected (mi)", 0.0, 10.0, 0.5)

if st.button('Predict Severity'):
    try:
        # Create a DataFrame with the EXACT columns the demo model expects
        input_data = pd.DataFrame({
            'Start_Lat': [lat],
            'Start_Lng': [lng],
            'Temperature(F)': [temperature],
            'Humidity(%)': [humidity],
            'Pressure(in)': [pressure],
            'Visibility(mi)': [visibility],
            'Wind_Speed(mph)': [wind_speed],
            'Precipitation(in)': [precip],
            'Distance(mi)': [distance]
        })

        # Match columns to model (just in case order differs)
        input_data = input_data[model.feature_names_in_]

        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)
        confidence = np.max(probability) * 100

        st.subheader("Prediction Result:")

        if prediction == 1:
            st.success(f"🟢 Severity 1: Very Minor (Confidence: {confidence:.1f}%)")
        elif prediction == 2:
            st.info(f"🔵 Severity 2: Minor / Moderate (Confidence: {confidence:.1f}%)")
        elif prediction == 3:
            st.warning(f"🟠 Severity 3: Serious (Confidence: {confidence:.1f}%)")
        else:
            st.error(f"🔴 Severity 4: FATAL (Confidence: {confidence:.1f}%)")

    except Exception as e:
        st.error(f"Error: {e}")