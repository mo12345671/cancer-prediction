import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    page_icon="🌸",
    layout="centered",
)

# --- Define the feature names and their ranges from the dataset ---
# These are the 30 features used for prediction.
# The ranges are estimated based on typical values for this dataset.
feature_details = {
    "radius_mean": {"label": "Radius (Mean)", "min": 6.98, "max": 28.11, "default": 17.99},
    "texture_mean": {"label": "Texture (Mean)", "min": 9.71, "max": 39.28, "default": 10.38},
    "perimeter_mean": {"label": "Perimeter (Mean)", "min": 43.79, "max": 188.5, "default": 122.8},
    "area_mean": {"label": "Area (Mean)", "min": 143.5, "max": 2501.0, "default": 1001.0},
    "smoothness_mean": {"label": "Smoothness (Mean)", "min": 0.05, "max": 0.16, "default": 0.118},
    "compactness_mean": {"label": "Compactness (Mean)", "min": 0.019, "max": 0.345, "default": 0.277},
    "concavity_mean": {"label": "Concavity (Mean)", "min": 0.0, "max": 0.426, "default": 0.300},
    "concave points_mean": {"label": "Concave Points (Mean)", "min": 0.0, "max": 0.201, "default": 0.147},
    "symmetry_mean": {"label": "Symmetry (Mean)", "min": 0.106, "max": 0.304, "default": 0.241},
    "fractal_dimension_mean": {"label": "Fractal Dimension (Mean)", "min": 0.05, "max": 0.097, "default": 0.078},
    "radius_se": {"label": "Radius (SE)", "min": 0.11, "max": 2.87, "default": 0.95},
    "texture_se": {"label": "Texture (SE)", "min": 0.36, "max": 4.88, "default": 0.9},
    "perimeter_se": {"label": "Perimeter (SE)", "min": 0.76, "max": 21.98, "default": 8.58},
    "area_se": {"label": "Area (SE)", "min": 6.8, "max": 542.2, "default": 153.4},
    "smoothness_se": {"label": "Smoothness (SE)", "min": 0.0017, "max": 0.031, "default": 0.0064},
    "compactness_se": {"label": "Compactness (SE)", "min": 0.002, "max": 0.135, "default": 0.046},
    "concavity_se": {"label": "Concavity (SE)", "min": 0.0, "max": 0.396, "default": 0.04},
    "concave points_se": {"label": "Concave Points (SE)", "min": 0.0, "max": 0.053, "default": 0.01},
    "symmetry_se": {"label": "Symmetry (SE)", "min": 0.007, "max": 0.078, "default": 0.02},
    "fractal_dimension_se": {"label": "Fractal Dimension (SE)", "min": 0.0009, "max": 0.03, "default": 0.004},
    "radius_worst": {"label": "Radius (Worst)", "min": 7.93, "max": 36.04, "default": 25.38},
    "texture_worst": {"label": "Texture (Worst)", "min": 12.02, "max": 49.54, "default": 17.33},
    "perimeter_worst": {"label": "Perimeter (Worst)", "min": 50.41, "max": 251.2, "default": 184.6},
    "area_worst": {"label": "Area (Worst)", "min": 185.2, "max": 4254.0, "default": 2019.0},
    "smoothness_worst": {"label": "Smoothness (Worst)", "min": 0.07, "max": 0.223, "default": 0.162},
    "compactness_worst": {"label": "Compactness (Worst)", "min": 0.027, "max": 1.058, "default": 0.665},
    "concavity_worst": {"label": "Concavity (Worst)", "min": 0.0, "max": 1.252, "default": 0.711},
    "concave points_worst": {"label": "Concave Points (Worst)", "min": 0.0, "max": 0.291, "default": 0.265},
    "symmetry_worst": {"label": "Symmetry (Worst)", "min": 0.156, "max": 0.664, "default": 0.46},
    "fractal_dimension_worst": {"label": "Fractal Dimension (Worst)", "min": 0.055, "max": 0.207, "default": 0.118}
}

# --- Function to load the model and scaler ---
@st.cache_resource
def load_resources():
    """Loads the pre-trained model and scaler using joblib."""
    try:
        model = joblib.load('model.sav')
        scaler = joblib.load('scaler.sav')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: 'model.sav' or 'scaler.sav' not found. Please ensure both files are in the same directory as this script.")
        return None, None

# Load the model and scaler
model, scaler = load_resources()

# --- Main app structure ---
st.title("Breast Cancer Diagnosis Prediction")
st.markdown("This app predicts whether a tumor is **Benign (B)** or **Malignant (M)** based on its characteristics. Adjust the sliders below to see the prediction change.")

if model and scaler:
    # Use columns to create a two-column layout for better organization
    cols = st.columns(2)
    user_inputs = {}

    # Create input widgets dynamically in a two-column layout
    col_index = 0
    for feature_name, details in feature_details.items():
        with cols[col_index % 2]:
            user_inputs[feature_name] = st.slider(
                details["label"],
                min_value=details["min"],
                max_value=details["max"],
                value=details["default"],
                step=(details["max"] - details["min"]) / 100,
                format="%.3f"
            )
        col_index += 1

    # Button to make a prediction
    if st.button("Predict Diagnosis", type="primary"):
        # Create a DataFrame from the user inputs
        input_data = pd.DataFrame([user_inputs])

        # Scale the input data using the loaded scaler
        scaled_input = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Display the result
        st.markdown("---")
        st.subheader("Prediction Result:")
        predicted_class = "Malignant" if prediction[0] == "M" else "Benign"
        
        # Display the final prediction with a colored box
        if predicted_class == "Malignant":
            st.error(f"The model predicts the tumor is **{predicted_class}**.")
        else:
            st.success(f"The model predicts the tumor is **{predicted_class}**.")

        # Display confidence scores
        st.write("---")
        st.subheader("Prediction Confidence:")
        
        # Determine the confidence scores for each class
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_).T.reset_index()
        proba_df.columns = ['Diagnosis', 'Confidence']
        proba_df['Confidence'] = proba_df['Confidence'] * 100
        
        st.dataframe(
            proba_df,
            column_order=('Diagnosis', 'Confidence'),
            hide_index=True,
            use_container_width=True,
        )
