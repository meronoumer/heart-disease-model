# streamlit_app.py
# ------------------------------------------------------------
# Heart Disease Model Demo App
# - Optional audio: compute many common audio features expected by your pipeline
# - Gender UI uses 'M'/'F', with safe fallback to numeric if pipeline needs it
# - Smoker UI uses 1/0
# - Aligns to artifact feature_cols, fills missing with NaN (imputer should handle)
# ------------------------------------------------------------

from __future__ import annotations

import io
import re
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Optional audio deps
try:
    import librosa
    import librosa.display  # noqa
except Exception:
    librosa = None

# Quiet hmmlearn noise if it shows up via artifacts
warnings.filterwarnings("ignore", message="Some rows of transmat_", module="hmmlearn")
warnings.filterwarnings("ignore", message="invalid value encountered", module="hmmlearn")

# -------------------------- App Config --------------------------
st.set_page_config(
    page_title="Heart Disease Model — Demo",
    page_icon="🫀",
    layout="wide",
)

DEFAULT_LABELS = ["AS", "AR", "MR", "MS", "N"]
DEFAULT_SEQ_KEY = "file_key"
DEFAULT_PATIENT_KEY = "patient_id_x"
DEFAULT_META_COLS = [
    "Unnamed: 0", "patient_id_x", "file_key", "audio_filename_base",
    "Age", "Gender", "Smoker", "Lives"
]
DEFAULT_LABEL_COLS = ["AS", "AR", "MR", "MS", "N"]

# -------------------------- Small helpers --------------------------
def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    m = x.mean()
    s = x.std() + 1e-12
    return float(((x - m) ** 3).mean() / (s ** 3))

def parse_mfcc_dev_str(s: str) -> List[float]:
    if pd.isna(s):
        return []
    content = str(s).strip().lstrip("[").rstrip("]")
    return [float(x) for x in content.split()] if content else []

def expand_mfcc_dev_column(df: pd.DataFrame, col: str = "mfcc devation") -> pd.DataFrame:
    # Parse existing "mfcc devation" column into mfcc_dev_* columns if present
    if col not in df.columns:
        return df
    parsed = df[col].apply(parse_mfcc_dev_str)
    if len(parsed) == 0 or len(parsed.iloc[0]) == 0:
        return df.drop(columns=[col])
    n_feats = len(parsed.iloc[0])
    mfcc_cols = [f"mfcc_dev_{i}" for i in range(n_feats)]
    df_mfcc = pd.DataFrame(parsed.tolist(), columns=mfcc_cols, index=df.index)
    return pd.concat([df.drop(columns=[col]), df_mfcc], axis=1)

def align_features(X: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    missing = [c for c in feature_cols if c not in X.columns]
    extra = [c for c in X.columns if c not in feature_cols]
    for c in missing:
        X[c] = np.nan
    X = X[feature_cols]
    return X, missing, extra

def guess_feature_cols_from_artifact(model_art):
    if isinstance(model_art, dict):
        if "feature_cols" in model_art:
            return list(model_art["feature_cols"])
        inner = model_art.get("pipeline") or model_art.get("model") or model_art.get("clf")
        if hasattr(inner, "feature_names_in_"):
            return list(inner.feature_names_in_)
    if hasattr(model_art, "feature_names_in_"):
        return list(model_art.feature_names_in_)
    return None

def guess_labels_from_artifact(model_art) -> List[str]:
    if isinstance(model_art, dict) and "labels" in model_art:
        return list(model_art["labels"])
    return DEFAULT_LABELS

def predict_proba_from_artifact(model_art, X: pd.DataFrame) -> np.ndarray:
    # Dictionary bundle?
    if isinstance(model_art, dict):
        if "predict_proba" in model_art and callable(model_art["predict_proba"]):
            return np.asarray(model_art["predict_proba"](X))
        for key in ("pipeline", "model", "clf", "stacked"):
            est = model_art.get(key)
            if est is not None and hasattr(est, "predict_proba"):
                return _normalize_proba(est.predict_proba(X))
    # Bare estimator/pipeline
    if hasattr(model_art, "predict_proba"):
        return _normalize_proba(model_art.predict_proba(X))
    raise RuntimeError("Loaded artifact does not expose predict_proba.")

def _normalize_proba(proba) -> np.ndarray:
    proba = np.asarray(proba)
    # Multi-label pipeline sometimes returns list-of-arrays
    if isinstance(proba, list):
        cols = []
        for arr in proba:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                cols.append(arr[:, 1])
            else:
                cols.append(arr.ravel())
        return np.column_stack(cols)
    # HMM-like (n, classes, 2) -> take pos class
    if proba.ndim == 3:
        return np.transpose(proba[:, :, 1])
    if proba.ndim == 1:
        return proba.reshape(-1, 1)
    return proba

def human_prob(p: float) -> str:
    return f"{100.0 * float(p):.1f}%"

# --------- Dataset-specific encodings ---------
def apply_dataset_encodings(df: pd.DataFrame, gender_val: str | None = None, smoker_val: int | None = None) -> pd.DataFrame:
    if gender_val is not None:
        df["Gender"] = str(gender_val).upper().strip()[:1]  # 'M' or 'F'
    if smoker_val is not None:
        df["Smoker"] = int(smoker_val)
    return df

def expand_or_map_categoricals(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # One-hot for Gender if expected
    if any(c.startswith("Gender_") for c in feature_cols):
        oh_targets = [c for c in feature_cols if c.startswith("Gender_")]
        for c in oh_targets:
            df[c] = 0
        if "Gender" in df.columns:
            tgt = f"Gender_{df['Gender'].iloc[0]}"
            if tgt in oh_targets:
                df[tgt] = 1
        if "Gender" not in feature_cols and "Gender" in df.columns:
            df.drop(columns=["Gender"], inplace=True, errors="ignore")
    # One-hot for Smoker if expected
    if any(c.startswith("Smoker_") for c in feature_cols):
        oh_targets = [c for c in feature_cols if c.startswith("Smoker_")]
        for c in oh_targets:
            df[c] = 0
        if "Smoker" in df.columns:
            v = int(df["Smoker"].iloc[0])
            candidates = [f"Smoker_{v}", "Smoker_Yes" if v == 1 else "Smoker_No"]
            for tgt in candidates:
                if tgt in oh_targets:
                    df[tgt] = 1
                    break
        if "Smoker" not in feature_cols and "Smoker" in df.columns:
            df.drop(columns=["Smoker"], inplace=True, errors="ignore")
    return df

def coerce_numerics_except_expected_categoricals(X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # Keep object for raw categoricals only if model expects them as strings
    keep_as_object = set()
    for cand in ("Gender", "Lives"):
        if cand in feature_cols and cand in X.columns:
            keep_as_object.add(cand)
    for c in X.columns:
        if c not in keep_as_object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

def force_numeric_categoricals_if_needed(X: pd.DataFrame) -> pd.DataFrame:
    """Fallback if model complains about strings: map common categoricals to numeric codes."""
    X = X.copy()
    if "Gender" in X.columns and X["Gender"].dtype == object:
        X["Gender"] = X["Gender"].map({"M": 1, "F": 0}).fillna(0).astype(float)
    # Simple ordinal mapping for Lives if present and object
    if "Lives" in X.columns and X["Lives"].dtype == object:
        ord_map = {"Rural": 0.0, "Suburban": 1.0, "Urban": 2.0}
        X["Lives"] = X["Lives"].map(ord_map).fillna(1.0).astype(float)
    # Any other unexpected object columns -> factorize to stable integers
    for c in X.columns:
        if X[c].dtype == object:
            codes, _ = pd.factorize(X[c].astype(str))
            X[c] = codes.astype(float)
    return X

# -------------------------- Audio feature extraction --------------------------
def compute_audio_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> Dict[str, float]:
    """Compute a set of common features whose names match your training columns where possible."""
    feats: Dict[str, float] = {}

    # Zero Crossing Rate (mean)
    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        feats["Mean Zero Crossing Rate"] = float(zcr.mean())
    except Exception:
        feats["Mean Zero Crossing Rate"] = np.nan

    # RMS stats
    try:
        rms = librosa.feature.rms(y=y).ravel()
        feats["Mean RMS"] = float(rms.mean())
        feats["Standard Dev. RMS"] = float(rms.std())
        feats["Skewness RMS"] = _skew(rms)
    except Exception:
        feats["Mean RMS"] = feats["Standard Dev. RMS"] = feats["Skewness RMS"] = np.nan

    # Spectral centroid/bandwidth
    try:
        sc = librosa.feature.spectral_centroid(y=y, sr=sr).ravel()
        feats["Mean Spectral Centroid"] = float(sc.mean())
    except Exception:
        feats["Mean Spectral Centroid"] = np.nan
    try:
        sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr).ravel()
        feats["Mean Spectral Bandwidth"] = float(sbw.mean())
    except Exception:
        feats["Mean Spectral Bandwidth"] = np.nan

    # Spectral contrast (avg over bands)
    try:
        scontrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats["Mean Spectral Contrast"] = float(scontrast.mean())
    except Exception:
        feats["Mean Spectral Contrast"] = np.nan

    # Mel spectrogram stats
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        feats["mean mel spectogram"] = float(mel.mean())  # keep misspelling to match column
        feats["mel spectrogram deviation"] = float(mel.std())
    except Exception:
        feats["mean mel spectogram"] = feats["mel spectrogram deviation"] = np.nan

    # CQT stats
    try:
        C = np.abs(librosa.cqt(y=y, sr=sr))
        feats["CQT Mean"] = float(C.mean())
        feats["CQT Std"] = float(C.std())
        feats["CQT Skew"] = _skew(C.ravel())
    except Exception:
        feats["CQT Mean"] = feats["CQT Std"] = feats["CQT Skew"] = np.nan

    # Spectral flux (approx)
    try:
        S = np.abs(librosa.stft(y=y))
        S = S / (S.sum(axis=0, keepdims=True) + 1e-12)
        flux = np.sqrt(((np.diff(S, axis=1)) ** 2).sum(axis=0))
        feats["spectralFlux_sma3_amean"] = float(flux.mean())
        feats["spectralFlux_sma3_stddevNorm"] = float(flux.std())
    except Exception:
        feats["spectralFlux_sma3_amean"] = feats["spectralFlux_sma3_stddevNorm"] = np.nan

    # MFCC features
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=max(13, n_mfcc))
        feats["mfcc length"] = float(mfcc.shape[1])
        # Means / stddevs for first 4 MFCCs to match typical columns
        for i in range(4):
            feats[f"mfcc{i+1}_sma3_amean"] = float(mfcc[i, :].mean())
            feats[f"mfcc{i+1}_sma3_stddevNorm"] = float(mfcc[i, :].std())
        # Deviation (std) per-coefficient as mfcc_dev_i
        dev = np.std(mfcc, axis=1)
        for i in range(dev.shape[0]):
            feats[f"mfcc_dev_{i}"] = float(dev[i])
    except Exception:
        # still include the MFCC names if they might be required
        feats["mfcc length"] = np.nan

    # Features we can't compute easily here (leave NaN; imputer should handle)
    for impossible in [
        "loudness_sma3_amean", "loudness_sma3_stddevNorm",
        "loudness_sma3_percentile20.0", "loudness_sma3_percentile50.0",
        "loudness_sma3_percentile80.0", "loudness_sma3_pctlrange0-2",
        "loudness_sma3_meanRisingSlope", "loudness_sma3_stddevRisingSlope",
        "loudness_sma3_meanFallingSlope", "loudness_sma3_stddevFallingSlope",
        "spectralFluxUV_sma3nz_amean", "F1amplitudeLogRelF0_sma3nz_amean",
        "F2amplitudeLogRelF0_sma3nz_amean", "F3amplitudeLogRelF0_sma3nz_amean",
        "alphaRatioUV_sma3nz_amean", "hammarbergIndexUV_sma3nz_amean",
        "slopeUV0-500_sma3nz_amean", "slopeUV500-1500_sma3nz_amean",
        "loudnessPeaksPerSec", "MeanUnvoicedSegmentLength",
        "equivalentSoundLevel_dBp",
    ]:
        feats.setdefault(impossible, np.nan)

    return feats

# -------------------------- Caching --------------------------
@st.cache_resource(show_spinner=False)
def load_model_resource(path: str):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# -------------------------- UI --------------------------
st.title("🫀 Heart Disease Model — Interactive Demo")
st.markdown(
    "Enter **patient inputs** and optionally upload an **audio file**. "
    "The app computes many audio features your model expects. "
    "**Note:** Educational use only."
)

with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model file path", "models/final_stacked_classifier_model.pkl")
    dataset_path = st.text_input("Dataset CSV (optional)", "extracted_features_df.csv")
    labels_input = st.text_input("Label names (comma-separated)", "AS,AR,MR,MS,N")
    user_labels = [x.strip() for x in labels_input.split(",") if x.strip()] or DEFAULT_LABELS

# Load resources
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
                thresholds_from_artifact = dict(model_artifact["thresholds"])
        else:
            st.warning("Provide a valid model path.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

dataset_df = None
if dataset_path.strip():
    try:
        dataset_df = load_csv_cached(dataset_path.strip())
    except Exception as e:
        st.warning(f"Could not load dataset CSV: {e}")

LABELS = labels_from_artifact or user_labels or DEFAULT_LABELS

tab_manual, tab_dataset, tab_csv, tab_about = st.tabs(
    ["🧑‍⚕️ Manual Entry", "📁 Pick from Dataset", "📦 Upload CSV", "ℹ️ About"]
)

# -------------------------- Manual Entry --------------------------
with tab_manual:
    st.subheader("Patient Inputs (manual) + Optional Audio")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        age = st.number_input("Age", min_value=0, max_value=120, value=60, step=1)
    with colB:
        gender = st.radio("Gender", options=["M", "F"], index=0, horizontal=True)
    with colC:
        smoker_flag = st.radio("Smoker (1=yes, 0=no)", options=[1, 0], index=1, horizontal=True)
    with colD:
        lives = st.selectbox("Lives (region)", options=["Urban", "Suburban", "Rural"], index=0)

    st.markdown("**Optional:** Upload an audio file to extract features.")
    if librosa is None:
        st.warning("Audio feature extraction requires `librosa`. Please add it to requirements.")
        audio_file = None
    else:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            audio_file = st.file_uploader("Audio file (wav/mp3/flac/ogg/m4a)", type=["wav", "mp3", "flac", "ogg", "m4a"])
        with col2:
            n_mfcc = st.slider("n_mfcc", min_value=8, max_value=40, value=13, step=1)
        with col3:
            sr_target = st.number_input("Sample rate (Hz)", min_value=4000, max_value=48000, value=22050, step=1000)

    def make_manual_row_df() -> pd.DataFrame:
        row: Dict[str, object] = {
            "Age": age,
            "Gender": gender,      # 'M'/'F' (may be converted later if needed)
            "Smoker": smoker_flag, # 1/0
            "Lives": lives,
        }

        # If audio present, compute feature set
        if librosa is not None and audio_file is not None:
            try:
                audio_bytes = audio_file.read()
                y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr_target, mono=True)
                audio_feats = compute_audio_features(y, sr, n_mfcc=n_mfcc)
                row.update(audio_feats)
            except Exception as e:
                st.error(f"Audio processing failed: {e}")

        return pd.DataFrame([row])

    if st.button("Run prediction", type="primary"):
        try:
            row_df = make_manual_row_df()
            row_df = expand_mfcc_dev_column(row_df, col="mfcc devation")

            feat_cols = feature_cols_from_artifact
            if feat_cols is None:
                # Fallback guess (if artifact doesn't specify)
                feat_cols = [c for c in row_df.columns if c not in DEFAULT_LABEL_COLS]

            # Enforce dataset encodings (M/F, 1/0)
            row_df = apply_dataset_encodings(row_df, gender_val=gender, smoker_val=smoker_flag)
            # One-hot expansion if model expects that form
            row_df = expand_or_map_categoricals(row_df, feat_cols)

            # Align & numeric conversion (keep expected raw strings if needed)
            X_aligned, missing, extra = align_features(row_df.copy(), feat_cols)
            if missing:
                st.info(f"Missing features were set to NaN (pipeline should impute): {missing}")
            if extra:
                st.caption(f"Ignored extra columns: {extra}")

            X_aligned = coerce_numerics_except_expected_categoricals(X_aligned, feat_cols)

            # Try prediction; if it complains about strings, fallback-encode categoricals to numeric and retry.
            try:
                with st.spinner("Scoring..."):
                    proba = predict_proba_from_artifact(model_artifact, X_aligned)
            except Exception as e:
                msg = str(e)
                if re.search(r"could not convert string to float", msg, flags=re.I):
                    X_numeric = force_numeric_categoricals_if_needed(X_aligned)
                    with st.spinner("Scoring (with numeric fallback)..."):
                        proba = predict_proba_from_artifact(model_artifact, X_numeric)
                else:
                    raise

            proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
            thresholds = thresholds_from_artifact or {lab: 0.5 for lab in LABELS}

            st.success("Prediction complete.")
            cols = st.columns(len(LABELS))
            for j, lab in enumerate(LABELS):
                p = proba[0, j] if proba.ndim == 2 else proba[0]
                dec = "POSITIVE" if p >= float(thresholds.get(lab, 0.5)) else "NEGATIVE"
                with cols[j]:
                    st.metric(label=f"{lab}", value=human_prob(p), delta=dec)

            with st.expander("Raw probabilities"):
                st.write(pd.DataFrame([proba[0]], columns=LABELS))

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------- Pick from Dataset --------------------------
with tab_dataset:
    st.subheader("Pick from Dataset")
    if dataset_df is None:
        st.info("Provide a dataset path in the sidebar to use this mode.")
    else:
        if DEFAULT_SEQ_KEY not in dataset_df.columns:
            st.error(f"Expected '{DEFAULT_SEQ_KEY}' column in dataset.")
        else:
            keys = sorted(dataset_df[DEFAULT_SEQ_KEY].astype(str).unique().tolist())
            selected_key = st.selectbox("Select a file_key", options=keys)

            if st.button("Predict for selected case"):
                try:
                    sub = dataset_df[dataset_df[DEFAULT_SEQ_KEY].astype(str) == selected_key]
                    if sub.empty:
                        st.error("No rows found for the selected key.")
                    else:
                        # Aggregate multiple frames to one row of numeric means
                        num_cols = [c for c in sub.columns if c not in (DEFAULT_META_COLS + DEFAULT_LABEL_COLS)]
                        g = sub[num_cols].select_dtypes(include=[np.number])
                        row_df = (g.mean(axis=0, skipna=True).to_frame().T
                                  if not g.empty else sub.iloc[[0]].drop(columns=DEFAULT_LABEL_COLS, errors="ignore"))

                        # Try to carry over simple categoricals if present
                        for c in ("Age", "Gender", "Smoker", "Lives"):
                            if c in sub.columns and c not in row_df.columns:
                                row_df[c] = sub[c].iloc[0]

                        row_df = expand_mfcc_dev_column(row_df, col="mfcc devation")

                        feat_cols = feature_cols_from_artifact or [c for c in row_df.columns if c not in DEFAULT_LABEL_COLS]

                        # Normalize encodings
                        if "Gender" in row_df.columns:
                            row_df["Gender"] = row_df["Gender"].astype(str).str.upper().map(lambda x: "M" if x.startswith("M") else "F")
                        if "Smoker" in row_df.columns:
                            row_df["Smoker"] = pd.to_numeric(row_df["Smoker"], errors="coerce").fillna(0).astype(int)

                        row_df = expand_or_map_categoricals(row_df, feat_cols)

                        X, missing, extra = align_features(row_df.copy(), feat_cols)
                        if missing:
                            st.info(f"Missing features set to NaN: {missing}")
                        if extra:
                            st.caption(f"Ignored extras: {extra}")

                        X = coerce_numerics_except_expected_categoricals(X, feat_cols)

                        try:
                            with st.spinner("Scoring..."):
                                proba = predict_proba_from_artifact(model_artifact, X)
                        except Exception as e:
                            if re.search(r"could not convert string to float", str(e), flags=re.I):
                                X = force_numeric_categoricals_if_needed(X)
                                with st.spinner("Scoring (with numeric fallback)..."):
                                    proba = predict_proba_from_artifact(model_artifact, X)
                            else:
                                raise

                        proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
                        thresholds = thresholds_from_artifact or {lab: 0.5 for lab in LABELS}

                        st.success("Prediction complete.")
                        cols = st.columns(len(LABELS))
                        for j, lab in enumerate(LABELS):
                            p = proba[0, j] if proba.ndim == 2 else proba[0]
                            dec = "POSITIVE" if p >= float(thresholds.get(lab, 0.5)) else "NEGATIVE"
                            with cols[j]:
                                st.metric(label=f"{lab}", value=human_prob(p), delta=dec)

                        with st.expander("Raw probabilities"):
                            st.write(pd.DataFrame([proba[0]], columns=LABELS))
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# -------------------------- Upload CSV --------------------------
with tab_csv:
    st.subheader("Upload CSV (single or batch)")
    st.write("If your CSV has multiple rows per `file_key`, you can aggregate to one row per key.")

    agg = st.checkbox("Aggregate by file_key (mean of numeric features)", value=True)
    up = st.file_uploader("CSV file", type=["csv"], key="csv_upl")

    if up is not None:
        try:
            df_u = pd.read_csv(up)
            df_u = expand_mfcc_dev_column(df_u, col="mfcc devation")

            if agg and DEFAULT_SEQ_KEY in df_u.columns:
                outs = []
                for key, grp in df_u.groupby(DEFAULT_SEQ_KEY, as_index=False):
                    num_cols = [c for c in grp.columns if c not in (DEFAULT_META_COLS + DEFAULT_LABEL_COLS)]
                    g = grp[num_cols].select_dtypes(include=[np.number])
                    row = (g.mean(axis=0, skipna=True).to_frame().T
                           if not g.empty else grp.iloc[[0]].drop(columns=DEFAULT_LABEL_COLS, errors="ignore"))
                    row[DEFAULT_SEQ_KEY] = str(key)
                    outs.append(row)
                df_u = pd.concat(outs, ignore_index=True)

            feat_cols = feature_cols_from_artifact or [c for c in df_u.columns if c not in DEFAULT_LABEL_COLS]

            # Normalize encodings if present
            if "Gender" in df_u.columns:
                df_u["Gender"] = df_u["Gender"].astype(str).str.upper().map(lambda x: "M" if x.startswith("M") else "F")
            if "Smoker" in df_u.columns:
                df_u["Smoker"] = pd.to_numeric(df_u["Smoker"], errors="coerce").fillna(0).astype(int)

            df_u = expand_or_map_categoricals(df_u, feat_cols)

            X, missing, extra = align_features(df_u.copy(), feat_cols)
            if missing:
                st.info(f"Missing features created as NaN: {missing}")
            if extra:
                st.caption(f"Ignored extras: {extra}")

            X = coerce_numerics_except_expected_categoricals(X, feat_cols)

            if st.button("Run CSV prediction", type="primary"):
                try:
                    with st.spinner("Scoring..."):
                        proba = predict_proba_from_artifact(model_artifact, X)
                except Exception as e:
                    if re.search(r"could not convert string to float", str(e), flags=re.I):
                        X = force_numeric_categoricals_if_needed(X)
                        with st.spinner("Scoring (with numeric fallback)..."):
                            proba = predict_proba_from_artifact(model_artifact, X)
                    else:
                        raise

                proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
                proba_df = pd.DataFrame(proba, columns=LABELS)
                out_df = pd.concat([df_u.reset_index(drop=True), proba_df], axis=1)

                thresholds = thresholds_from_artifact or {lab: 0.5 for lab in LABELS}
                for lab in LABELS:
                    thr = float(thresholds.get(lab, 0.5))
                    out_df[f"{lab}_pred"] = (out_df[lab] >= thr).astype(int)

                st.success("Batch prediction complete.")
                st.dataframe(out_df.head(100), use_container_width=True)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"CSV processing/prediction failed: {e}")

# -------------------------- About --------------------------
with tab_about:
    st.subheader("About this app")
    st.markdown(
        """
        **Inputs & Encoding**
        - **Gender**: `'M'` / `'F'` in the UI. If your model expects numeric, the app will fallback to `M→1, F→0`.
        - **Smoker**: `1` (Yes) / `0` (No).
        - **Audio (optional)**: The app computes many features your model likely expects:
          Zero Crossing Rate, RMS mean/std/skew, Spectral Centroid/Bandwidth/Contrast,
          Mel spectrogram mean/std, CQT mean/std/skew, Spectral Flux, MFCC means/std for first 4 bands,
          `mfcc_dev_*`, and `mfcc length`. Some OpenSMILE-specific features remain NaN and should be imputed.

        **Feature Alignment**
        - If your model artifact provides `feature_cols`, the app aligns to them.
          Any missing columns are created as `NaN` so your imputer can fill them.

        **Disclaimer**  
        Not a medical device. For educational/research use only.
        """
    )
