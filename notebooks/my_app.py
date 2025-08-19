import streamlit as st
import pandas as pd
import joblib
import librosa as lib
import numpy as np
import scipy
import math

def extract_features_from_uploaded_file(uploaded_file):
    try:
        y, sr = lib.load(uploaded_file, sr=None)

        fmin = 50
        max_n_bands = int(math.log2((sr / 2) / fmin))
        if max_n_bands < 1:
            raise ValueError("Sampling rate too low for spectral contrast with given fmin")

        zero_crossings = lib.feature.zero_crossing_rate(y)
        rms_energy = lib.feature.rms(y=y).flatten()
        spectral_centroid = lib.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = lib.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = lib.feature.spectral_contrast(y=y, sr=sr, fmin=fmin, n_bands=max_n_bands)
        mfccs = lib.feature.mfcc(y=y, sr=sr)
        log_mel_spect = lib.feature.melspectrogram(y=y, sr=sr)

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
            "Std Mel Spectrogram": np.std(log_mel_spect),
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
        "Location": location_map[location],
    }


st.title("Heart Disease Detection Dashboard")
st.header("Patient Information")

age = st.slider("Age", 20, 100, 50, help="Patient's age in years")
smoker = st.selectbox("Are you a smoker?", ["Yes", "No"])

col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("Place of residence", ["Rural", "Urban"])
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])

st.header("Upload Data")
uploaded_file = st.file_uploader("Upload a PCG file", type=["wav", "mp3"])

st.write("Patient Information:", {
    "Age": age,
    "Gender": gender,
    "Smoker": smoker,
    "Location": location
})


if uploaded_file is not None:
    st.success("File uploaded successfully.")

    audio_features = extract_features_from_uploaded_file(uploaded_file)


    if audio_features is not None:
        metadata_features = encode_metadata(age, gender, smoker, location)
        all_features = {**audio_features, **metadata_features}

        input_df = pd.DataFrame([all_features])

        input_df = input_df.loc[:, ~input_df.columns.str.contains('^Unnamed')]


        feature_names = joblib.load("../models/feature_names.pkl")
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        # Load model and predict
        model = joblib.load("../models/final_classifier_model.pkl")

        if st.button("Run Prediction"):
            prediction = model.predict(input_df)[0]

            disease_labels = ["Aortic Stenosis", "Aortic Regurgitation", "Mitral Regurgitation", "Mitral Regurgitation", "Mitral Stenosis", "Normal"]
            predicted_diseases = [disease for disease, present in zip(disease_labels, prediction) if present == 1]

            if predicted_diseases:
                st.write("Predicted Diseases:")
                for d in predicted_diseases:
                    st.write(f"- {d}")
            else:
                st.write("No diseases predicted.")

