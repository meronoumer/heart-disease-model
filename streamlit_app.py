from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Use the NaN-safe loader + predictor from the shim we added earlier
from predict_safe_shim import predict_safe, load_artifacts_from_bundle

# Label order must match your training order
LABELS_FULL = [
    "Aortic Stenosis",
    "Aortic Regurgitation",
    "Mitral Regurgitation",
    "Mitral Stenosis",
    "Normal",
]

def most_likely_from_probs(probs) -> tuple[int, float]:
    """
    Return (best_idx, best_p_present) for a **single sample**.
    Accepts numpy arrays or sklearn's list-of-arrays from MultiOutput.
    """
    import numpy as np

    # List-of-arrays case (common for MultiOutputClassifier)
    if isinstance(probs, list):
        p_present = []
        for p in probs:
            a = np.asarray(p)
            # expect shape (1, n_classes). Use last column as "present".
            if a.ndim == 1:
                # (2,) → binary
                p_present.append(float(a[-1]))
            elif a.ndim == 2:
                p_present.append(float(a[0, -1]))
            else:
                # fallback: flatten
                p_present.append(float(a.reshape(-1)[-1]))
        p_present = np.asarray(p_present)

    else:
        a = np.asarray(probs)
        # expect (n_labels, 2) for a single sample
        if a.ndim == 2:
            if a.shape[1] >= 2:
                p_present = a[:, 1]
            else:
                p_present = a[:, 0]
        elif a.ndim == 3:
            # (n_samples, n_labels, n_classes) → use sample 0
            if a.shape[2] >= 2:
                p_present = a[0, :, 1]
            else:
                p_present = a[0, :, 0]
        elif a.ndim == 1:
            # degenerate: a single binary prob for one label
            p_present = np.array([a[-1]])
        else:
            # last-resort flatten
            flat = a.reshape(-1)
            p_present = np.array([flat[-1]])

    best_idx = int(np.argmax(p_present))
    best_p = float(p_present[best_idx])
    return best_idx, best_p

st.set_page_config(page_title="Heart Disease Model – Simple Predictor", layout="centered")


# =========================
# ---- Audio features  ----
# =========================
def _try_import_librosa():
    try:
        import librosa  # type: ignore
        return librosa
    except Exception:
        return None


def extract_audio_features(
    file_bytes: bytes,
    feature_cols: List[str],
    sr_target: int = 22050,
    n_mfcc: int = 20,
) -> Dict[str, float]:
    """
    Best-effort audio feature extractor.
    - Tries librosa (if available) for MP3/WAV/FLAC/OGG/M4A, etc.
    - Returns a dict of features using *common names*; extras are fine (they get dropped later).
    - If extraction fails, returns {} so the model imputes medians for audio features.
    """
    L: Dict[str, float] = {}
    librosa = _try_import_librosa()
    if librosa is None:
        # No audio backend available; silently fall back
        st.info("Audio libraries not detected; proceeding without audio features (medians will be used).")
        return L

    try:
        # Load audio from memory (BytesIO works with librosa/audioread)
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr_target, mono=True)
        if y.size == 0:
            return L

        # Basic normalization
        y = librosa.util.normalize(y)

        # --- Features ---
        # MFCCs (means)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_means = mfcc.mean(axis=1)

        # Zero-crossing rate (mean)
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())

        # RMS energy (mean)
        rms = float(librosa.feature.rms(y=y).mean())

        # Spectral centroid (mean)
        spec_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())

        # Populate flexible keys; extras get dropped later
        # MFCC names: 'mfcc_01'...'mfcc_20' are common in your project
        for i, val in enumerate(mfcc_means, start=1):
            L[f"mfcc_{i:02d}"] = float(val)
            # Also drop in a couple alt keys in case your training used different naming
            L[f"mfcc{i:02d}"] = float(val)
            L[f"mfcc{i}"] = float(val)

        # Common single-value names
        for name, value in [
            ("zcr", zcr),
            ("zcr_mean", zcr),
            ("rms", rms),
            ("rms_mean", rms),
            ("spectral_centroid", spec_centroid),
            ("spec_centroid", spec_centroid),
            ("spec_centroid_mean", spec_centroid),
        ]:
            L[name] = value

        # Only keep keys that exist in feature_cols (optional pruning here; extras would be dropped later anyway)
        # L = {k: v for k, v in L.items() if k in feature_cols}

        return L
    except Exception:
        st.warning("Couldn’t read audio (codec/backend missing). Continuing without audio features.")
        return {}


# =========================
# ---- Demographic map ----
# =========================
def demographics_to_features(
    feature_cols: List[str],
    gender_choice: str,  # "Female" or "Male"
    smoker_choice: str,  # "No" or "Yes"
    age_value: float,
) -> Dict[str, float]:
    """
    Map UI inputs to feature columns *if those columns exist* in the trained model.
    We set only columns we find; the rest will be median-imputed.
    """
    d: Dict[str, float] = {}
    male = 1.0 if gender_choice == "Male" else 0.0
    smoker = 1.0 if smoker_choice == "Yes" else 0.0

    # Age
    for candidate in ["Age", "age"]:
        if candidate in feature_cols:
            d[candidate] = float(age_value)
            break

    # Gender / Sex (we prefer a single numeric column; if absent, skip)
    for candidate in ["Gender", "gender", "sex", "is_male", "male"]:
        if candidate in feature_cols:
            d[candidate] = male
            break

    # Smoker
    for candidate in ["Smoker", "smoker", "is_smoker"]:
        if candidate in feature_cols:
            d[candidate] = smoker
            break

    return d


# =========================
# ---------  UI  ---------
# =========================
def main():
    st.title("Cardiac Murmur / Disease Classifier")
    st.caption("Enter demographics, optionally upload a heartbeat audio file, and get a prediction.")

    # Load bundle once (cached inside the shim)
    bundle_path = "artifacts/final_stacked_classifier_model_bundle.pkl"
    try:
        art = load_artifacts_from_bundle(bundle_path)
    except Exception as e:
        st.error(f"Could not load model bundle: {e}")
        st.stop()

    feature_cols: List[str] = art["feature_cols"]

    # --- Inputs ---
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
    with c2:
        smoker = st.selectbox("Smoker", options=["No", "Yes"], index=0)
    with c3:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

    audio_file = st.file_uploader(
        "Heartbeat audio (optional)", type=["wav", "mp3", "ogg", "flac", "m4a", "aac"]
    )

    if st.button("Predict", use_container_width=True):
        # Build a raw feature dict from demographics
        raw_dict = demographics_to_features(feature_cols, gender, smoker, age)

        # If an audio file was provided, try to add audio features
        if audio_file is not None:
            audio_bytes = audio_file.read()
            audio_feats = extract_audio_features(audio_bytes, feature_cols)
            raw_dict.update(audio_feats)

        # Create a one-row DataFrame for inference
        raw_df = pd.DataFrame([raw_dict])

        # Route through our NaN-safe predictor with bundle medians
        try:
            y_pred, probs, X_infer, fcols, needs_manual = predict_safe(
                raw_features_df=raw_df,
                prefer="median",  # we always use medians; no UI toggle
                bundle_path=bundle_path,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # --- Results ---
        st.subheader("Result")

        if probs is not None:
            best_idx, best_p = most_likely_from_probs(probs)
            label = LABELS_FULL[best_idx] if best_idx < len(LABELS_FULL) else f"Label {best_idx}"
            st.markdown(f"**{label}** is **{best_p*100:.1f}%** likely.")
        else:
            # Fallback if the model doesn’t expose predict_proba
            # We pick the first positive prediction if any; confidence not available.
            y = np.asarray(y_pred).ravel().tolist()
            try:
                best_idx = y.index(1)
                label = LABELS_FULL[best_idx] if best_idx < len(LABELS_FULL) else f"Label {best_idx}"
                st.markdown(f"**{label}** is **(confidence unavailable)**.")
            except ValueError:
                st.markdown("**No condition predicted present** (confidence unavailable).")
        with st.expander("Details (preprocessing)", expanded=False):
            missing = [c for c in fcols if c not in raw_df.columns]
            extra = [c for c in raw_df.columns if c not in fcols]
            st.write("Audio used:", audio_file is not None and len(raw_dict) > 3)
            st.write("Expected features:", len(fcols))
            if missing:
                st.info(f"Imputed {len(missing)} missing feature(s).")
            if extra:
                st.warning(f"Dropped {len(extra)} unexpected feature(s).")
            st.write("Any NaNs left in model input?:", bool(pd.isna(X_infer).any().any()))


if __name__ == "__main__":
    main()
