import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import joblib  # for compatibility with different save methods

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Cancer Prediction App", layout="centered")
st.title("🧬 Cancer Prediction using Pre-trained Best Model")

# -----------------------------
# Safe Loader (pickle or joblib)
# -----------------------------
def safe_load(path):
    """Try loading with pickle first, then joblib."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"❌ Failed to load {path}: {e}")
            return None

# -----------------------------
# Load the best model
# -----------------------------
model_path = "best_model.sav"
model = safe_load(model_path) if os.path.exists(model_path) else None

if model:
    st.success("✅ Best Model loaded successfully!")
else:
    st.error("❌ Could not load 'best_model.sav'. Please check the file.")

# -----------------------------
# Feature Names (10 only)
# -----------------------------
feature_names = [
    'mean radius', 
    'mean texture', 
    'mean perimeter', 
    'mean area', 
    'mean smoothness',
    'radius error', 
    'texture error', 
    'perimeter error', 
    'area error', 
    'smoothness error'
]

# -----------------------------
# Default realistic values (for user convenience)
# -----------------------------
defaults = {
    'mean radius': 14.0,
    'mean texture': 19.0,
    'mean perimeter': 90.0,
    'mean area': 600.0,
    'mean smoothness': 0.1,
    'radius error': 0.4,
    'texture error': 1.2,
    'perimeter error': 3.0,
    'area error': 40.0,
    'smoothness error': 0.01
}

# -----------------------------
# Prediction Section
# -----------------------------
if model is not None:

    option = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

    # ------------------ Manual Input ------------------
    if option == "Manual Input":
        st.subheader("🔢 Enter Features Manually")

        input_values = []
        for feature in feature_names:
            val = st.number_input(f"{feature}", value=defaults.get(feature, 0.0))
            input_values.append(val)

        input_data = np.array([input_values])

        if st.button("Predict"):
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.error("⚠️ Malignant (Cancerous Tumor)")
            else:
                st.success("✅ Benign (Non-Cancerous Tumor)")

            # Probability (if supported)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0][1]
                st.write(f"Probability of Malignancy: **{prob:.2f}**")

    # ------------------ CSV Upload ------------------
    elif option == "Upload CSV":
        uploaded_csv = st.file_uploader("📂 Upload CSV with Features", type=["csv"])
        
        # Template Download Button
        st.download_button(
            "⬇️ Download CSV Template",
            data=pd.DataFrame(columns=feature_names).to_csv(index=False).encode("utf-8"),
            file_name="cancer_features_template.csv",
            mime="text/csv"
        )

        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            st.write("📊 Uploaded Data Preview:", df.head())

            # Validate Columns
            if list(df.columns) != feature_names:
                st.error("❌ CSV columns do not match the expected feature names.")
            else:
                df_values = df.values  # raw values

                if st.button("Predict from CSV"):
                    preds = model.predict(df_values)
                    df["Prediction"] = preds

                    if hasattr(model, "predict_proba"):
                        df["Malignancy_Prob"] = model.predict_proba(df_values)[:, 1]

                    st.write("✅ Predictions Done!")
                    st.dataframe(df)

                    # Download results
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
