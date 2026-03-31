# app.py

import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Bean Classifier", layout="centered")

st.title("🌱 Dry Bean Type Classifier")
st.write("Enter bean characteristics to predict its type")

# ---- Input Fields ----
Area = st.number_input("Area")
Perimeter = st.number_input("Perimeter")
MajorAxisLength = st.number_input("Major Axis Length")
MinorAxisLength = st.number_input("Minor Axis Length")
AspectRatio = st.number_input("Aspect Ratio")
Eccentricity = st.number_input("Eccentricity")
ConvexArea = st.number_input("Convex Area")
EquivDiameter = st.number_input("Equivalent Diameter")
Extent = st.number_input("Extent")
Solidity = st.number_input("Solidity")
Roundness = st.number_input("Roundness")
Compactness = st.number_input("Compactness")
ShapeFactor1 = st.number_input("Shape Factor 1")
ShapeFactor2 = st.number_input("Shape Factor 2")
ShapeFactor3 = st.number_input("Shape Factor 3")
ShapeFactor4 = st.number_input("Shape Factor 4")

# ---- Prediction ----
if st.button("Predict Bean Type"):
    
    features = np.array([[Area, Perimeter, MajorAxisLength, MinorAxisLength,
                          AspectRatio, Eccentricity, ConvexArea, EquivDiameter,
                          Extent, Solidity, Roundness, Compactness,
                          ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4]])
    
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    
    st.success(f"🌾 Predicted Bean Type: {prediction[0]}")