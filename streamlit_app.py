# streamlit_app.py
# ------------------------------------------------------------
# Heart Disease Model Demo App
#
# Features:
# - Configurable model loader (no hard-coded paths)
# - Use data from repo CSV, user-uploaded CSV (single/batch), or uploaded audio
# - Parses "mfcc devation" -> mfcc_dev_* columns to match training schema
# - Aligns features to model's expected columns
# - Displays per-label probabilities and decisions (thresholded)
# - Batch scoring with downloadable results
# - Caching + robust error handling
#
# Suggested requirements.txt (example):
# streamlit
# pandas
# numpy
# scikit-learn
# joblib
# librosa
# soundfile
# hmmlearn   # if your artifact uses HMMs internally
#
# ------------------------------------------------------------

from __future__ import annotations

import io
import os
import sys
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Optional audio dependencies
try:
    import librosa
except Exception:
    librosa = None

# Quiet known noisy warnings from hmmlearn if your model bundles HMMs
warnings.filterwarnings("ignore", message="Some rows of transmat_", module="hmmlearn")
warnings.filterwarnings("ignore", message="invalid value encountered", module="hmmlearn")

# -------------------------- App Config --------------------------
st.set_page_config(
    page_title="Heart Disease Model — Demo",
    page_icon="🫀",
    layout="wide",
)

# Constants used in your training pipeline
DEFAULT_LABELS = ["AS", "AR", "MR", "MS", "N"]
DEFAULT_SEQ_KEY = "file_key"
DEFAULT_PATIENT_KEY = "patient_id_x"
DEFAULT_META_COLS = [
    "Unnamed: 0", "patient_id_x", "file_key", "audio_filename_base",
    "Age", "Gender", "Smoker", "Lives"
]
DEFAULT_LABEL_COLS = ["AS", "AR", "MR", "MS", "N"]

# -------------------------- Utility Functions --------------------------
def parse_mfcc_dev_str(s: str) -> List[float]:
    """Parse a string like '[0.1 0.2 0.3 ...]' into a list of floats."""
    if pd.isna(s):
        return []
    content = str(s).strip().lstrip("[").rstrip("]")
    return [float(x) for x in content.split()] if content else []

def expand_mfcc_dev_column(df: pd.DataFrame, col: str = "mfcc devation") -> pd.DataFrame:
    """If 'mfcc devation' exists, expand it into mfcc_dev_* columns."""
    if col not in df.columns:
        return df

    parsed = df[col].apply(parse_mfcc_dev_str)
    if len(parsed) == 0 or len(parsed.iloc[0]) == 0:
        # Nothing to expand; drop the text column to avoid confusion
        return df.drop(columns=[col])

    n_feats = len(parsed.iloc[0])
    mfcc_cols = [f"mfcc_dev_{i}" for i in range(n_feats)]
    df_mfcc = pd.DataFrame(parsed.tolist(), columns=mfcc_cols, index=df.index)
    out = pd.concat([df.drop(columns=[col]), df_mfcc], axis=1)
    return out

def align_features(
    X: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Align X's columns to the required feature_cols."""
    missing = [c for c in feature_cols if c not in X.columns]
    extra = [c for c in X.columns if c not in feature_cols]

    # Create missing columns with NaN; downstream pipelines should handle imputation
    for c in missing:
        X[c] = np.nan

    X = X[feature_cols]
    return X, missing, extra

def guess_feature_cols_from_artifact(model_art):
    """Try to read feature columns from the artifact in a robust way."""
    # If artifact is a dict-like bundle (your earlier pattern)
    if isinstance(model_art, dict):
        if "feature_cols" in model_art:
            return list(model_art["feature_cols"])
        # Sometimes a nested pipeline might exist:
        inner = model_art.get("pipeline") or model_art.get("model") or model_art.get("clf")
        if hasattr(inner, "feature_names_in_"):
            return list(inner.feature_names_in_)

    # If it's a scikit estimator/pipeline
    if hasattr(model_art, "feature_names_in_"):
        return list(model_art.feature_names_in_)

    # Unknown; fallback to None (we'll try to infer from dataset)
    return None

def guess_labels_from_artifact(model_art) -> List[str]:
    """Try to read labels from the artifact; fallback to DEFAULT_LABELS."""
    if isinstance(model_art, dict):
        if "labels" in model_art and isinstance(model_art["labels"], (list, tuple)):
            return list(model_art["labels"])

    # For multi-output classifiers, there may not be a simple .classes_ list
    # Default to your known labels
    return DEFAULT_LABELS

def predict_proba_from_artifact(model_art, X: pd.DataFrame) -> np.ndarray:
    """
    Try to produce (n_samples, n_labels) probabilities.
    Supports:
    - scikit estimators with predict_proba (single-label or multi-label/OvR)
    - custom dict bundles exposing a callable 'predict_proba' or 'pipeline'
    """
    # If dict bundle with callable
    if isinstance(model_art, dict):
        # Common storage patterns:
        if "predict_proba" in model_art and callable(model_art["predict_proba"]):
            return model_art["predict_proba"](X)

        # Pipeline/estimator inside dict
        for key in ("pipeline", "model", "clf"):
            if key in model_art and hasattr(model_art[key], "predict_proba"):
                est = model_art[key]
                proba = est.predict_proba(X)
                # predict_proba can be list of arrays (for multi-output OvR style)
                if isinstance(proba, list):
                    # Stack per-output probabilities of positive class
                    cols = []
                    for arr in proba:
                        arr = np.asarray(arr)
                        if arr.ndim == 2 and arr.shape[1] >= 2:
                            cols.append(arr[:, 1])
                        else:
                            cols.append(arr.ravel())
                    return np.column_stack(cols)
                proba = np.asarray(proba)
                if proba.ndim == 3:
                    # Some wrappers return (n_outputs, n_samples, n_classes)
                    # Reduce to positive class per output
                    return np.transpose(proba[:, :, 1])
                if proba.ndim == 2 and proba.shape[1] > 1:
                    # Single multi-class -> return as-is
                    return proba
                # Otherwise treat as binary
                return proba.reshape(-1, 1)

        # If no direct estimator, try simple weighted logic (if saved)
        if "stacked" in model_art and hasattr(model_art["stacked"], "predict_proba"):
            proba = model_art["stacked"].predict_proba(X)
            return np.asarray(proba)

    # If bare estimator/pipeline
    if hasattr(model_art, "predict_proba"):
        proba = model_art.predict_proba(X)
        if isinstance(proba, list):
            cols = []
            for arr in proba:
                arr = np.asarray(arr)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    cols.append(arr[:, 1])
                else:
                    cols.append(arr.ravel())
            return np.column_stack(cols)
        proba = np.asarray(proba)
        if proba.ndim == 3:
            return np.transpose(proba[:, :, 1])
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba
        return proba.reshape(-1, 1)

    raise RuntimeError("Loaded artifact does not expose a usable predict_proba interface.")

def aggregate_group_rows_to_single_vector(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Convert possibly many rows per file_key into a single feature row.
    - Parses 'mfcc devation' (if present) from the FIRST non-null and expands to mfcc_dev_*
    - Averages other numeric columns
    """
    # Work on a copy
    g = grp.copy()

    # Expand MFCC deviation if present
    g = expand_mfcc_dev_column(g, col="mfcc devation")

    # Pick numeric columns (skip obvious meta/labels)
    cols_to_skip = set(DEFAULT_META_COLS + DEFAULT_LABEL_COLS)
    num_cols = [c for c in g.columns if c not in cols_to_skip and pd.api.types.is_numeric_dtype(g[c])]

    if not num_cols:
        # Fallback: just take first row (after expansion)
        return g.iloc[[0]].drop(columns=[c for c in g.columns if c in DEFAULT_LABEL_COLS], errors="ignore")

    # Aggregate by mean for numeric features
    agg = g[num_cols].mean(axis=0, skipna=True).to_frame().T
    return agg

def human_prob(p: float) -> str:
    return f"{100.0 * float(p):.1f}%"

def df_download_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# -------------------------- Caching --------------------------
@st.cache_resource(show_spinner=False)
def load_model_resource(path: str):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# -------------------------- Sidebar: Configuration --------------------------
st.title("🫀 Heart Disease Model — Interactive Demo")
st.markdown(
    "This app lets you run predictions with your **merged ensemble model**. "
    "You can pick a case from your dataset, upload your own CSV, or even upload an audio file "
    "to extract MFCC-based features on the fly.\n\n"
    "**Note:** This tool is for **educational/research** use only and **not** a medical device."
)

with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input(
        "Model file path",
        value="final_stacked_classifier_model.pkl",
        help="Path to your model artifact. You can replace this with any accessible path.",
    )
    dataset_path = st.text_input(
        "Dataset CSV (optional)",
        value="extracted_features_df.csv",
        help="Used for 'Pick from dataset' mode. Leave blank if you won't use it.",
    )
    labels_input = st.text_input(
        "Label names (comma-separated)",
        value="AS,AR,MR,MS,N",
        help="Used for display and decisions if the artifact doesn't provide labels.",
    )
    user_labels = [x.strip() for x in labels_input.split(",") if x.strip()] or DEFAULT_LABELS

    st.caption("Tip: Update paths and labels here. The app reloads resources automatically when changed.")

# Load model (with error handling)
model_artifact = None
feature_cols_from_artifact = None
labels_from_artifact = None
thresholds_from_artifact: Optional[Dict[str, float]] = None

with st.spinner("Loading model..."):
    try:
        if model_path.strip():
            model_artifact = load_model_resource(model_path.strip())
            feature_cols_from_artifact = guess_feature_cols_from_artifact(model_artifact)
            labels_from_artifact = guess_labels_from_artifact(model_artifact)
            if isinstance(model_artifact, dict) and "thresholds" in model_artifact:
                # Expect thresholds as dict {label: float}
                thresholds_from_artifact = dict(model_artifact["thresholds"])
        else:
            st.warning("Please provide a valid model file path in the sidebar.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# Load dataset (optional)
dataset_df = None
if dataset_path.strip():
    try:
        dataset_df = load_csv_cached(dataset_path.strip())
    except Exception as e:
        st.warning(f"Could not load dataset CSV: {e}")

# Decide labels used for display
LABELS = labels_from_artifact or user_labels or DEFAULT_LABELS

# -------------------------- Tabs --------------------------
tab_predict, tab_batch, tab_audio, tab_about = st.tabs(["🔮 Predict", "📦 Batch", "🎙️ Audio", "ℹ️ About"])

# -------------------------- Predict (single) --------------------------
with tab_predict:
    st.subheader("Single Prediction")

    mode = st.radio(
        "Choose input source:",
        options=["Pick from dataset", "Upload a CSV row"],
        horizontal=True,
    )

    # Thresholds (can override)
    with st.expander("Decision Thresholds (optional)", expanded=False):
        thresholds = {}
        for lab in LABELS:
            default_thr = 0.5
            if thresholds_from_artifact and lab in thresholds_from_artifact:
                default_thr = float(thresholds_from_artifact[lab])
            thresholds[lab] = st.slider(f"Threshold for {lab}", 0.0, 1.0, float(default_thr), 0.01)

    if mode == "Pick from dataset":
        if dataset_df is None:
            st.info("Provide a dataset path in the sidebar to use this mode.")
        else:
            # Prepare options for file_key
            if DEFAULT_SEQ_KEY not in dataset_df.columns:
                st.error(f"Expected '{DEFAULT_SEQ_KEY}' column in dataset.")
            else:
                keys = sorted(dataset_df[DEFAULT_SEQ_KEY].astype(str).unique().tolist())
                selected_key = st.selectbox("Select a file_key", options=keys)

                if st.button("Run prediction", type="primary"):
                    try:
                        sub = dataset_df[dataset_df[DEFAULT_SEQ_KEY].astype(str) == selected_key]
                        if sub.empty:
                            st.error("No rows found for the selected key.")
                        else:
                            # Aggregate to a single feature row
                            row_df = aggregate_group_rows_to_single_vector(sub)

                            # If artifact provides feature columns, align
                            feat_cols = feature_cols_from_artifact
                            if feat_cols is None:
                                # If unknown, attempt to derive from dataset by removing meta/labels
                                feat_cols = [c for c in row_df.columns
                                             if c not in (DEFAULT_META_COLS + DEFAULT_LABEL_COLS)]

                            X, missing, extra = align_features(row_df.copy(), feat_cols)
                            if missing:
                                st.warning(f"Missing columns were created as NaN (model pipeline should impute): {missing}")
                            if extra:
                                st.caption(f"Ignored extra columns: {extra}")

                            # Predict
                            with st.spinner("Scoring..."):
                                proba = predict_proba_from_artifact(model_artifact, X)
                                proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)

                            # Present results
                            st.success("Prediction complete.")
                            cols = st.columns(len(LABELS))
                            for j, lab in enumerate(LABELS):
                                p = proba[0, j] if proba.ndim == 2 else proba[0]
                                dec = "POSITIVE" if p >= thresholds[lab] else "NEGATIVE"
                                with cols[j]:
                                    st.metric(label=f"{lab}", value=human_prob(p), delta=dec)

                            with st.expander("Raw probabilities"):
                                st.write(pd.DataFrame([proba[0]], columns=LABELS))
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    else:  # Upload a CSV row
        st.write("Upload a CSV containing a single row with required features. "
                 "If it includes 'mfcc devation', it will be parsed automatically.")
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            try:
                df_u = pd.read_csv(up)
                if len(df_u) != 1:
                    st.warning("This mode expects exactly 1 row. If you need multiple rows, use the Batch tab.")
                # Expand mfcc devation if present
                df_u = expand_mfcc_dev_column(df_u, col="mfcc devation")
                # Align features
                feat_cols = feature_cols_from_artifact
                if feat_cols is None:
                    # Best effort: remove meta & labels from uploaded row
                    feat_cols = [c for c in df_u.columns if c not in (DEFAULT_META_COLS + DEFAULT_LABEL_COLS)]
                X, missing, extra = align_features(df_u.copy(), feat_cols)
                if missing:
                    st.warning(f"Missing columns were created as NaN: {missing}")
                if extra:
                    st.caption(f"Ignored extra columns: {extra}")

                if st.button("Run prediction", type="primary"):
                    with st.spinner("Scoring..."):
                        proba = predict_proba_from_artifact(model_artifact, X)
                        proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
                    st.success("Prediction complete.")
                    cols = st.columns(len(LABELS))
                    for j, lab in enumerate(LABELS):
                        p = proba[0, j] if proba.ndim == 2 else proba[0]
                        dec = "POSITIVE" if p >= thresholds[lab] else "NEGATIVE"
                        with cols[j]:
                            st.metric(label=f"{lab}", value=human_prob(p), delta=dec)
                    with st.expander("Raw probabilities"):
                        st.write(pd.DataFrame([proba[0]], columns=LABELS))
            except Exception as e:
                st.error(f"Upload/parse failed: {e}")

# -------------------------- Batch --------------------------
with tab_batch:
    st.subheader("Batch Scoring (CSV)")
    st.write(
        "Upload a CSV with one or more rows. If it contains multiple rows per `file_key`, "
        "you can choose to aggregate them into a single row per key."
    )
    agg = st.checkbox("Aggregate by file_key (mean of numeric features)", value=True)
    upb = st.file_uploader("Batch CSV", type=["csv"], key="batch_csv")

    if upb is not None:
        try:
            df_b = pd.read_csv(upb)
            # Expand mfcc devation if present
            df_b = expand_mfcc_dev_column(df_b, col="mfcc devation")

            if agg:
                if DEFAULT_SEQ_KEY not in df_b.columns:
                    st.error(f"Aggregation requested but '{DEFAULT_SEQ_KEY}' column not found.")
                else:
                    # Build one vector per key
                    outs = []
                    for key, grp in df_b.groupby(DEFAULT_SEQ_KEY, as_index=False):
                        row = aggregate_group_rows_to_single_vector(grp)
                        row[DEFAULT_SEQ_KEY] = str(key)
                        outs.append(row)
                    df_b = pd.concat(outs, ignore_index=True)

            feat_cols = feature_cols_from_artifact
            if feat_cols is None:
                feat_cols = [c for c in df_b.columns if c not in (DEFAULT_META_COLS + DEFAULT_LABEL_COLS)]

            X, missing, extra = align_features(df_b.copy(), feat_cols)
            if missing:
                st.warning(f"Missing columns created as NaN: {missing}")
            if extra:
                st.caption(f"Ignored extra columns: {extra}")

            if st.button("Run batch prediction", type="primary"):
                with st.spinner("Scoring batch..."):
                    proba = predict_proba_from_artifact(model_artifact, X)
                    proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
                st.success("Batch prediction complete.")
                proba_df = pd.DataFrame(proba, columns=LABELS)
                out_df = pd.concat([df_b.reset_index(drop=True), proba_df], axis=1)

                # Add decisions
                for lab in LABELS:
                    thr = thresholds_from_artifact.get(lab, 0.5) if thresholds_from_artifact else 0.5
                    out_df[f"{lab}_pred"] = (out_df[lab] >= thr).astype(int)

                st.dataframe(out_df.head(100), use_container_width=True)
                df_download_button(out_df, "Download predictions CSV", "predictions.csv")

        except Exception as e:
            st.error(f"Batch scoring failed: {e}")

# -------------------------- Audio --------------------------
with tab_audio:
    st.subheader("Predict from Audio (optional)")
    st.write(
        "Upload an audio file (e.g., WAV/MP3). The app will compute MFCC-based summary features "
        "to approximate the 'mfcc devation' vector used in training."
    )
    if librosa is None:
        st.warning("Audio feature extraction requires `librosa`. Please add it to requirements.")
    else:
        audio_file = st.file_uploader(
            "Audio file",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key="audio_upl",
        )
        n_mfcc = st.slider("Number of MFCC coefficients", 8, 40, 13, 1,
                           help="This should match what was used in training; 13 is common.")
        sr_target = st.number_input("Target sample rate (Hz)", min_value=4000, max_value=48000, value=22050, step=1000)

        # Thresholds (audio tab can use same thresholds as predict tab)
        with st.expander("Decision Thresholds (optional)", expanded=False):
            thresholds_audio = {}
            for lab in LABELS:
                default_thr = 0.5
                if thresholds_from_artifact and lab in thresholds_from_artifact:
                    default_thr = float(thresholds_from_artifact[lab])
                thresholds_audio[lab] = st.slider(f"Threshold for {lab}", 0.0, 1.0, float(default_thr), 0.01, key=f"thr_audio_{lab}")

        def extract_mfcc_dev_from_audio(file_bytes: bytes, sr: int, n_mfcc_: int) -> pd.DataFrame:
            """Return a single-row DataFrame with mfcc_dev_* columns."""
            y, sr_loaded = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True)
            # Compute MFCCs: shape (n_mfcc, n_frames)
            mfcc = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=n_mfcc_)
            # Deviation across frames (std deviation)
            dev = np.std(mfcc, axis=1)
            data = {f"mfcc_dev_{i}": dev[i] for i in range(n_mfcc_)}
            return pd.DataFrame([data])

        if audio_file is not None and st.button("Run audio prediction", type="primary"):
            try:
                audio_bytes = audio_file.read()
                row_df = extract_mfcc_dev_from_audio(audio_bytes, sr_target, n_mfcc)
                # Align to model features
                feat_cols = feature_cols_from_artifact
                if feat_cols is None:
                    # If unknown, use available mfcc_dev_* columns only
                    feat_cols = [c for c in row_df.columns if c.startswith("mfcc_dev_")]

                X, missing, extra = align_features(row_df.copy(), feat_cols)
                if missing:
                    st.info("Some required features were missing and set to NaN; "
                            "ensure your model pipeline includes an imputer.")
                with st.spinner("Scoring audio..."):
                    proba = predict_proba_from_artifact(model_artifact, X)
                    proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
                st.success("Prediction complete from audio.")

                cols = st.columns(len(LABELS))
                for j, lab in enumerate(LABELS):
                    p = proba[0, j] if proba.ndim == 2 else proba[0]
                    dec = "POSITIVE" if p >= thresholds_audio[lab] else "NEGATIVE"
                    with cols[j]:
                        st.metric(label=f"{lab}", value=human_prob(p), delta=dec)
                with st.expander("Raw probabilities"):
                    st.write(pd.DataFrame([proba[0]], columns=LABELS))
            except Exception as e:
                st.error(f"Audio prediction failed: {e}")

# -------------------------- About --------------------------
with tab_about:
    st.subheader("About this app")
    st.markdown(
        """
        **Purpose**  
        This app demonstrates a heart disease prediction model. It is intended for **education and research** only.

        **How it works**  
        - Loads a merged/stacked model from a user-provided path (default shown in sidebar).
        - Inputs can be taken from your dataset, an uploaded CSV, or from audio (which is converted to MFCC-based features).
        - Features are aligned to what the model expects; missing features are set to NaN (your model/pipeline should impute).

        **Tips**  
        - Keep your CSV schema consistent with training.
        - If predictions fail, check the feature list and thresholds in the model artifact.
        - For audio-based predictions, ensure the MFCC setup matches training as closely as practical.

        **Disclaimer**  
        This application does **not** provide medical advice and must **not** be used for clinical decisions.
        """
    )
