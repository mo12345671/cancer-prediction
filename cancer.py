import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Cancer Prediction App", layout="centered")
st.title("🧬 Cancer Prediction using Pre-trained Model")

# -----------------------------
# Load Pickle Model from Device
# -----------------------------
model_path = "model (1).pkl"  # make sure this file is in the same folder as app.py

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model Loaded Successfully from device!")
else:
    st.error("❌ Pickle file not found. Please make sure 'best_model.pkl' is in the app folder.")

# -----------------------------
# Feature Names (top 10 selected features)
# Replace this list with the actual features your model was trained on
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error'
]

# -----------------------------
# Prediction Mode
# -----------------------------
if 'model' in locals():
    option = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

    # Manual Input
    if option == "Manual Input":
        st.subheader("🔢 Enter Features Manually")

        input_values = []
        for feature in feature_names:
            val = st.number_input(f"{feature}", value=0.0)
            input_values.append(val)

        input_data = np.array([input_values])

        if st.button("Predict"):
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.error("⚠️ Malignant (Cancerous Tumor)")
            else:
                st.success("✅ Benign (Non-Cancerous Tumor)")

    # CSV Input
    elif option == "Upload CSV":
        uploaded_csv = st.file_uploader("Upload CSV with Features", type=["csv"])
        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            st.write("📂 Uploaded Data Preview:", df.head())

            if st.button("Predict from CSV"):
                preds = model.predict(df)
                df["Prediction"] = preds
                st.write("✅ Predictions Done!")
                st.dataframe(df)

                # Download results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

