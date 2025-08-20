# streamlit_app.py
import os
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ===========================
# Page / Style
# ===========================
st.set_page_config(
    page_title="Cardiovascular Diagnostic Aid",
    page_icon="🫀",
    layout="centered"
)

HIDE_RAW_WARNINGS = True
if HIDE_RAW_WARNINGS:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

PRIMARY_LABELS = ["AS", "AR", "MR", "MS", "N"]  # fallback if we cannot infer labels

# ===========================
# Utilities
# ===========================
def _safe_load(path: str):
    """Load a pickled model bundle with joblib, fallback to pickle."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        # fallback to pickle if joblib fails
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

@st.cache_resource(show_spinner=True)
def load_model_bundle(model_path: str):
    """Return (model_or_pipeline, feature_cols, labels) from a variety of saved formats."""
    bundle = _safe_load(model_path)

    # common cases:
    # 1) dict with 'pipeline' (or 'model') and 'feature_cols'
    # 2) direct sklearn Pipeline/Estimator with feature_names saved
    pipe = None
    feature_cols = None
    labels = None

    if isinstance(bundle, dict):
        pipe = bundle.get("pipeline") or bundle.get("model") or bundle.get("estimator") or bundle
        feature_cols = bundle.get("feature_cols")
        # optional saved labels
        labels = bundle.get("labels") or bundle.get("classes") or bundle.get("class_names")
    else:
        pipe = bundle

    # Try to infer feature columns if not provided
    if feature_cols is None:
        # (a) many sklearn transformers store feature_names_in_
        if hasattr(pipe, "feature_names_in_"):
            feature_cols = list(pipe.feature_names_in_)
        else:
            # (b) best-effort: try ColumnTransformer in a Pipeline
            try:
                if hasattr(pipe, "named_steps"):
                    for step_name, step in pipe.named_steps.items():
                        if hasattr(step, "feature_names_in_"):
                            feature_cols = list(step.feature_names_in_)
                            break
            except Exception:
                pass

    if feature_cols is None:
        st.warning(
            "Could not infer expected feature columns from the model. "
            "Predictions may fail if inputs don't match training schema."
        )
        feature_cols = []  # keep going, but we’ll fall back to whatever the user provides

    # Try to infer labels if not provided
    if labels is None:
        # OneVsRestClassifier etc. often expose classes_
        if hasattr(pipe, "classes_"):
            labels = list(getattr(pipe, "classes_"))
        else:
            labels = PRIMARY_LABELS

    return pipe, feature_cols, labels

def ensure_feature_schema(row_dict: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Create a 1-row DataFrame with exactly the expected columns.
    Missing cols -> NaN; unexpected keys -> dropped."""
    if not feature_cols:
        # If we couldn't infer expected columns, use what we have.
        X = pd.DataFrame([row_dict])
    else:
        X = pd.DataFrame([{c: row_dict.get(c, np.nan) for c in feature_cols}])

    # Coerce to numeric when appropriate (keep categorical as strings if your pipeline encodes them)
    for c in X.columns:
        if X[c].dtype == object and c not in ("Gender", "Smoker"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

def show_missing_features(missing_cols: list[str], reason: str, *, compact_if_audio_only=True):
    """User-friendly notice about imputed features."""
    if not missing_cols:
        return

    audio_like = {
        "Mean Zero Crossing Rate","Mean RMS","Standard Dev. RMS","Skewness RMS",
        "Mean Spectral Centroid","Mean Spectral Bandwidth","Mean Spectral Contrast",
        "mfcc length","mean mel spectogram","mel spectrogram deviation",
        "CQT Mean","CQT Std","CQT Skew",
        "loudness_sma3_amean","loudness_sma3_stddevNorm",
        "loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0",
        "loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope",
        "loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope",
        "spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm",
        "mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm",
        "mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm",
        "F1amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_amean",
        "alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean",
        "slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean",
        "loudnessPeaksPerSec","MeanUnvoicedSegmentLength","equivalentSoundLevel_dBp","Unnamed: 0"
    }
    non_audio_missing = [c for c in missing_cols if c not in audio_like]

    if compact_if_audio_only and not non_audio_missing:
        st.caption("No audio provided — audio features will be imputed by the pipeline.")
        return

    st.info(f"{len(missing_cols)} features will be imputed ({reason}).")
    with st.expander("Show missing feature names"):
        st.code(", ".join(missing_cols), language="text")

def try_predict_proba(model, X):
    """Return dict: label -> probability, or None if not available."""
    try:
        proba = model.predict_proba(X)
    except Exception:
        return None

    # proba can be:
    # - array (n_samples, n_classes)
    # - list of arrays for multilabel OneVsRest [(n_samples,2), ...]
    if isinstance(proba, list):
        # One array per class; use positive-class probability at column 1
        out = np.array([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in proba]).T
        return out
    elif isinstance(proba, np.ndarray):
        return proba
    else:
        return None

def get_label_names(model, fallback_labels=PRIMARY_LABELS):
    """Try to extract class labels in a robust way."""
    if hasattr(model, "classes_"):
        cls = list(model.classes_)
        # For multilabel OneVsRest, classes_ is a list of arrays; try to flatten
        if len(cls) and isinstance(cls[0], (list, np.ndarray)):
            try:
                cls = [str(c[1]) if len(c) > 1 else str(c[0]) for c in cls]
            except Exception:
                cls = fallback_labels
        else:
            cls = [str(c) for c in cls]
        return cls
    return fallback_labels

def compute_audio_features(file: io.BytesIO) -> dict:
    """(Optional) Extract audio features to match training names.
    This is a lightweight placeholder. If you don't compute them,
    the pipeline imputer will fill them and still allow predictions."""
    # For now, return {} to rely on imputation.
    # You can later implement librosa-based extraction mapping to your exact training names.
    return {}

# ===========================
# App UI
# ===========================
st.title("🫀 Cardiovascular Disease Diagnostic Aid")
st.markdown(
    "Upload patient attributes and (optionally) an audio file. "
    "This app uses your trained ensemble/stacked model to estimate label probabilities."
)

with st.sidebar:
    st.header("Settings")
    default_model_path = "models/final_stacked_classifier_model.pkl"
    model_path = st.text_input("Model file path", value=default_model_path, help="Path relative to the repository root.")
    show_expected_cols = st.checkbox("Show expected feature columns after loading", value=False)

# Patient inputs
st.header("Patient Information")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
with c2:
    gender = st.selectbox("Gender", options=["M", "F"], index=0, help="Model expects 'M' or 'F'")
with c3:
    smoker = st.selectbox("Smoker", options=[0, 1], index=0, help="0 = No, 1 = Yes")

st.header("Optional: Upload Audio")
audio_file = st.file_uploader("Upload heart sound/audio file", type=["wav", "mp3", "flac", "ogg"])

st.divider()
predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

# ===========================
# Inference
# ===========================
if predict_btn:
    try:
        with st.spinner("Loading model…"):
            model, feature_cols, labels = load_model_bundle(model_path)

        if show_expected_cols:
            st.subheader("Expected feature columns")
            if feature_cols:
                st.code("\n".join(feature_cols), language="text")
            else:
                st.caption("Model did not expose expected columns. Using provided inputs only.")

        # Build base row from tabular UI fields
        row = {
            "Age": age,
            "Gender": gender,        # already 'M' or 'F'
            "Smoker": int(smoker),   # 0/1 as requested
        }

        # Add audio features if provided & computed
        if audio_file is not None:
            try:
                row.update(compute_audio_features(audio_file))
                audio_reason = "audio uploaded, but some features may still be imputed"
            except Exception as e:
                st.warning(f"Audio feature extraction failed; proceeding with imputation. ({e})")
                audio_reason = "audio uploaded but features not extracted"
        else:
            audio_reason = "no audio uploaded"

        # Align schema exactly to model expectation
        X = ensure_feature_schema(row, feature_cols)

        # Friendly missing-features message
        if feature_cols:
            missing = [c for c in feature_cols if pd.isna(X.at[0, c])]
        else:
            # If we couldn't infer columns, there's nothing reliable to compare
            missing = []

        show_missing_features(missing, audio_reason)

        # Predict
        # First, try predict_proba
        proba = try_predict_proba(model, X)

        # Fallbacks if predict_proba not available
        if proba is None:
            # decision_function -> map to 0..1 via sigmoid for display
            if hasattr(model, "decision_function"):
                scores = np.atleast_2d(model.decision_function(X))
                proba = 1 / (1 + np.exp(-scores))
            else:
                # last resort: predict only
                preds = np.atleast_1d(model.predict(X))
                # Convert to pseudo prob for display
                proba = np.atleast_2d(preds.astype(float))

        # Build a nice output table
        # Many multilabel pipelines return list-of-arrays or (1, C)
        if isinstance(proba, list):
            # convert list-of-arrays to (1, C) positive class probabilities
            arr = np.array([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in proba]).T
        else:
            arr = np.array(proba)

        # Determine label names
        label_names = labels if labels else get_label_names(model, PRIMARY_LABELS)

        # If the shape doesn't match labels, fall back safely
        if arr.ndim == 2 and arr.shape[1] != len(label_names):
            # Try fallback labels
            label_names = [f"class_{i}" for i in range(arr.shape[1])]

        # Display results
        st.subheader("Predicted Probabilities")
        probs_df = pd.DataFrame(arr, columns=label_names)
        # Clip for display, handle any odd values
        probs_df = probs_df.clip(lower=0.0, upper=1.0)

        # Show as bars
        for col in probs_df.columns:
            st.write(f"**{col}**")
            st.progress(float(probs_df.iloc[0][col]))

        with st.expander("Show raw prediction table"):
            st.dataframe(probs_df.style.format("{:.3f}"), use_container_width=True)

        # Optional: show predicted labels (threshold 0.5 by default; adjust if you saved custom thresholds)
        pred_labels = [lbl for lbl, p in zip(probs_df.columns, probs_df.iloc[0].values) if p >= 0.5]
        st.success(f"Predicted label(s): {', '.join(pred_labels) if pred_labels else 'None ≥ 0.5'}")

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
