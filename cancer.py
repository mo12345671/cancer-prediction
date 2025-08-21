import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# -----------------------------
# Feature list
# -----------------------------
selected_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 
    'mean smoothness', 'mean compactness', 'mean concavity', 
    'mean concave points', 'mean symmetry', 'mean fractal dimension'
]

# -----------------------------
# Load model and scaler safely
# -----------------------------
@st.cache_resource
def load_model_and_scaler():
    BASE_DIR = os.path.dirname(__file__)  # Current folder of the script

    model_path = os.path.join(BASE_DIR, 'model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found: {scaler_path}")
        return None, None

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load fitted scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model_and_scaler()

if model is not None and scaler is not None:
    # -----------------------------
    # App title
    # -----------------------------
    st.title("Breast Cancer Prediction with Feature Effects")
    st.write("Enter the values for each feature below to predict if a case is Malignant or Benign.")

    # -----------------------------
    # Input form
    # -----------------------------
    inputs = []
    for feature in selected_features:
        value = st.number_input(
            feature,
            min_value=0.0,
            step=0.01,
            format="%.4f",
            key=feature.replace(' ', '_')
        )
        inputs.append(value)

    input_array = np.array([inputs])
    scaled_input = scaler.transform(input_array)

    # -----------------------------
    # Prediction and probability
    # -----------------------------
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0]
    result = 'Malignant' if prediction == 1 else 'Benign'

    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {result}")
    st.write(f"Probability of Benign: {prob[0]*100:.2f}%")
    st.write(f"Probability of Malignant: {prob[1]*100:.2f}%")

    # -----------------------------
    # Feature effect chart (Logistic Regression coefficients)
    # -----------------------------
    coefs = model.coef_[0]
    feature_df = pd.DataFrame({
        'Feature': selected_features,
        'Effect': coefs
    }).sort_values(by='Effect', ascending=True)

    st.subheader("Feature Effect on Prediction")
    st.bar_chart(feature_df.set_index('Feature'))
    st.write("Positive effect → pushes prediction toward Malignant, Negative effect → pushes toward Benign")
else:
    st.error("Model or scaler could not be loaded. Please check the file paths.")
