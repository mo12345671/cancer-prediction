import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# -----------------------------
# Features used for training
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
    BASE_DIR = os.path.dirname(__file__)

    model_path = os.path.join(BASE_DIR, 'model.sav')
    scaler_path = os.path.join(BASE_DIR, 'scaler.sav')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model or scaler file not found. Make sure 'model.sav' and 'scaler.sav' exist in the app folder.")
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_scaler()

if model is not None and scaler is not None:
    st.title("Breast Cancer Prediction with Feature Effects")
    st.write("Enter values for the following features:")

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

    if st.button("Predict"):
        input_array = np.array([inputs])

        # -----------------------------
        # Scale input correctly
        # -----------------------------
        scaled_input = scaler.transform(input_array)

        # -----------------------------
        # Prediction
        # -----------------------------
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0]
        result = 'Malignant' if prediction == 1 else 'Benign'

        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {result}")
        st.write(f"Probability of Benign: {prob[0]*100:.2f}%")
        st.write(f"Probability of Malignant: {prob[1]*100:.2f}%")

        # -----------------------------
        # Feature effect chart (Logistic Regression)
        # -----------------------------
        coefs = model.coef_[0]
        feature_df = pd.DataFrame({
            'Feature': selected_features,
            'Effect': coefs
        }).sort_values(by='Effect', ascending=True)

        st.subheader("Feature Effect on Prediction")
        st.bar_chart(feature_df.set_index('Feature'))
        st.write("Positive effect → pushes prediction toward Malignant, Negative effect → pushes toward Benign")
