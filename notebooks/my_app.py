import streamlit as st
import pandas as pd
import os
import joblib
import librosa as lib
import numpy as np
import scipy



# streamlit_app.py


def extract_features_from_uploaded_file(uploaded_file, patient_id="unknown"):
    import librosa as lib
    import scipy
    import numpy as np
    import pandas as pd
    import math

    try:
        # Load audio from uploaded file
        y, sr = lib.load(uploaded_file, sr=None)



        fmin = 50
        max_n_bands = int(math.log2((sr / 2) / fmin))

        print(f"sr: {sr}, nyquist: {sr/2}, max n_bands: {max_n_bands}")

        # Prevent edge case failure
        if max_n_bands < 1:
            raise ValueError("Sampling rate too low for spectral contrast with given fmin")

        



        # Feature extraction (same as before)
        zero_crossings = lib.feature.zero_crossing_rate(y)
        rms_energy = lib.feature.rms(y=y).flatten()
        spectral_centroid = lib.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = lib.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = lib.feature.spectral_contrast(y=y, sr=sr, fmin=fmin, n_bands=max_n_bands)        
        mfccs = lib.feature.mfcc(y=y, sr=sr)

        log_mel_spect = lib.feature.melspectrogram(y=y, sr=sr)
        # cqt = lib.cqt(y, sr=sr, n_bins=10)
        # cqt_db = lib.amplitude_to_db(np.abs(cqt))

        # Aggregate statistics
        features = {
            "Mean Zero Crossing Rate": np.mean(zero_crossings),
            "Mean RMS": np.mean(rms_energy),
            "Std RMS": np.std(rms_energy),
            "Skew RMS": scipy.stats.skew(rms_energy),
            "Mean Spectral Centroid": np.mean(spectral_centroid),
            "Mean Spectral Bandwidth": np.mean(spectral_bandwidth),
            "Mean Spectral Contrast": np.mean(spectral_contrast),
            "Mean MFCC": np.mean(mfccs),
            "Std MFCC": np.std(mfccs),
            "Mean Mel Spectrogram": np.mean(log_mel_spect),
            "Std Mel Spectrogram": np.std(log_mel_spect)
            # "Mean CQT": np.mean(cqt_db),
            # "Std CQT": np.std(cqt_db),
            # "Skew CQT": scipy.stats.skew(cqt_db.flatten()),
        }

        return features

    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def encode_metadata(age, gender, smoker, location):
    gender_map = {"Male": 0, "Female": 1}
    smoker_map = {"No": 0, "Yes": 1}
    location_map = {"Urban": 0, "Rural": 1}

    return {
        "Age": age,
        "Gender": gender_map[gender],
        "Smoker": smoker_map[smoker],
        "Location": location_map[location]
    }

#UI 
st.title("Heart Disease Detection Dashboard")



st.header("Patient Information")

# --- Feature Inputs with Sliders ---
age = st.slider("Age", 20, 100, 50,help = "Patient's age in years ")


smoker = st.selectbox(
   label = "Are you a smoker?",
   options = ["Yes","No"]
)


col1, col2= st.columns(2)

with col1:
    location = st.selectbox(
    label = "How would you describe your place of residence?",
    options = ["Rural","Urban"]
    )
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])

st.header("Upload Data")
uploaded_file = st.file_uploader("Upload a PCG file", type=["wav", "mp3"])


# Optional: You can add checkboxes, radios, selects, etc. too!

# Combine into feature vector
features = np.array([[age, gender, smoker,location]])

# Example prediction placeholder
# prediction = model.predict(features)[0]
# st.write(f"Predicted Risk: {'High' if prediction == 1 else 'Low'}")

# For now:
st.write("Patient Information:")

patient_info = {
    "Age": 36,
    "Gender": gender,
    "Smoker": smoker,
    "Location": location
}

st.write(patient_info)


if uploaded_file is not None:
    st.success("File uploaded successfully.")

    # Extract audio features
    audio_features = extract_features_from_uploaded_file(uploaded_file)

    if audio_features is not None:
        # Encode metadata
        metadata_features = encode_metadata(age, gender, smoker, location)

        # Combine dictionaries
        all_features = {**audio_features, **metadata_features}

        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([all_features])

        # Load model
        model = joblib.load("../models/final_stacked_classifier_model.pkl")

        # Make prediction
        prediction = model.predict(input_df)[0]

        st.write("Prediction:", prediction)
