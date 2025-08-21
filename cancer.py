import streamlit as st
import numpy as np
import pickle
import pandas as pd

# -----------------------------
# List of features
# -----------------------------
selected_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 
    'mean smoothness', 'mean compactness', 'mean concavity', 
    'mean concave points', 'mean symmetry', 'mean fractal dimension'
]

# -----------------------------
# Load model and scaler
# -----------------------------
@st.cache_resource
def load_model_and_scaler():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model,scaler = load_model_and_scaler()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Breast Cancer Prediction with Feature Effect")

# Input form
inputs = []
for feature in selected_features:
    value = st.number_input(feature, min_value=0.0, step=0.01, format="%.4f", key=feature)
    inputs.append(value)

input_array = np.array([inputs])
scaled_input = scaler.transform(input_array)

# Prediction
prediction = model.predict(scaled_input)[0]
prob = model.predict_proba(scaled_input)[0]
result = 'Malignant' if prediction == 1 else 'Benign'

st.write(f"**Prediction:** {result}")
st.write(f"Probability of Benign: {prob[0]*100:.2f}%")
st.write(f"Probability of Malignant: {prob[1]*100:.2f}%")

# -----------------------------



