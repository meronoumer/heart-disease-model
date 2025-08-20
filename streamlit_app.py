# streamlit_app.py
# ------------------------------------------------------------
# Heart Disease Model Demo App
# - Optional audio: compute many audio features expected by your pipeline
# - Auto-impute missing values (dataset means -> 0.0) for models like GB that reject NaN
# - Gender UI uses 'M'/'F', with safe numeric fallback if pipeline needs it
# - Smoker UI uses 1/0
# - Aligns to artifact feature_cols; extra columns are ignored
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
    # Dict bundle?
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
    if isinstance(proba, list):
        cols = []
        for arr in proba:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                cols.append(arr[:, 1])
            else:
                cols.append(arr.ravel())
        return np.column_stack(cols)
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
    keep_as_object = set()
    for cand in ("Gender", "Lives"):
        if cand in feature_cols and cand in X.columns:
            keep_as_object.add(cand)
    for c in X.columns:
        if c not in keep_as_object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

def force_numeric_categoricals_if_needed(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "Gender" in X.columns and X["Gender"].dtype == object:
        X["Gender"] = X["Gender"].map({"M": 1, "F": 0}).fillna(0).astype(float)
    if "Lives" in X.columns and X["Lives"].dtype == object:
        ord_map = {"Rural": 0.0, "Suburban": 1.0, "Urban": 2.0}
        X["Lives"] = X["Lives"].map(ord_map).fillna(1.0).astype(float)
    for c in X.columns:
        if X[c].dtype == object:
            codes, _ = pd.factorize(X[c].astype(str))
            X[c] = codes.astype(float)
    return X

# -------------------------- Audio feature extraction --------------------------
def compute_audio_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> Dict[str, float]:
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

    # Spectral contrast
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
        for i in range(4):
            feats[f"mfcc{i+1}_sma3_amean"] = float(mfcc[i, :].mean())
            feats[f"mfcc{i+1}_sma3_stddevNorm"] = float(mfcc[i, :].std())
        dev = np.std(mfcc, axis=1)
        for i in range(dev.shape[0]):
            feats[f"mfcc_dev_{i}"] = float(dev[i])
    except Exception:
        feats["mfcc length"] = np.nan

    # Fields we cannot compute without OpenSMILE -> leave NaN; imputer will fill
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

# -------------------------- NEW: Imputation helpers --------------------------
def compute_fill_values(feature_cols: List[str], dataset_df: Optional[pd.DataFrame]) -> pd.Series:
    """Mean of numeric columns from reference dataset; default 0.0 if not available."""
    fill = pd.Series(0.0, index=pd.Index(feature_cols, dtype=object))
    if dataset_df is not None:
        # try to use dataset means when column present & numeric
        for c in feature_cols:
            if c in dataset_df.columns:
                try:
                    fill[c] = pd.to_numeric(dataset_df[c], errors="coerce").mean()
                    if np.isnan(fill[c]):
                        fill[c] = 0.0
                except Exception:
                    fill[c] = 0.0
    # also add harmless default for common meta index
    if "Unnamed: 0" in fill.index and np.isnan(fill["Unnamed: 0"]):
        fill["Unnamed: 0"] = 0.0
    return fill

def impute_locally(X: pd.DataFrame, fill_values: pd.Series) -> pd.DataFrame:
    Z = X.copy()
    # Ensure numeric dtypes (objects -> numeric codes where needed)
    for c in Z.columns:
        if Z[c].dtype == object:
            codes, _ = pd.factorize(Z[c].astype(str))
            Z[c] = codes.astype(float)
    # Align fill_values
    fv = fill_values.reindex(Z.columns)
    Z = Z.fillna(fv)
    # Any remaining NaN -> 0.0
    Z = Z.fillna(0.0)
    return Z

def attempt_predict_with_fallbacks(model_artifact, X: pd.DataFrame,
                                   dataset_df: Optional[pd.DataFrame],
                                   thresholds_from_artifact: Optional[Dict[str, float]],
                                   LABELS: List[str]) -> Tuple[np.ndarray, Dict[str, float], Optional[str]]:
    """Try normal prediction; if errors mention strings or NaNs, apply fallbacks."""
    # 1) Try as-is
    try:
        proba = predict_proba_from_artifact(model_artifact, X)
        return np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0), (thresholds_from_artifact or {lab: 0.5 for lab in LABELS}), None
    except Exception as e1:
        msg1 = str(e1)

    # 2) If string conversion issue -> force numeric categoricals
    if re.search(r"could not convert string to float", msg1, flags=re.I):
        X2 = force_numeric_categoricals_if_needed(X)
        try:
            proba = predict_proba_from_artifact(model_artifact, X2)
            return np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0), (thresholds_from_artifact or {lab: 0.5 for lab in LABELS}), "Converted categoricals to numeric."
        except Exception as e2:
            msg2 = str(e2)
            # fall through to imputation if NaN complaint persists
            msg1 = msg1 + " | " + msg2

    # 3) If NaN complaint -> local imputation (dataset means -> 0.0)
    if re.search(r"Input X contains NaN", msg1, flags=re.I) or re.search(r"NaN", msg1, flags=re.I):
        fill_values = compute_fill_values(list(X.columns), dataset_df)
        X3 = impute_locally(X, fill_values)
        try:
            proba = predict_proba_from_artifact(model_artifact, X3)
            note = "Imputed missing values (dataset means, fallback 0.0)."
            return np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0), (thresholds_from_artifact or {lab: 0.5 for lab in LABELS}), note
        except Exception as e3:
            raise RuntimeError(f"Prediction failed even after imputation. Last error: {e3}") from e3

    # Otherwise fail with original error
    raise RuntimeError(f"Prediction failed: {msg1}")

# -------------------------- UI --------------------------
st.title("🫀 Heart Disease Model — Interactive Demo")
st.markdown(
    "Enter **patient inputs** and optionally upload an **audio file**. "
    "If audio is not provided, the app will **impute** audio features using dataset means or 0.0 "
    "so models like Gradient Boosting can still run. **Educational use only.**"
)

with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model file path", "models/final_stacked_classifier_model.pkl")
    dataset_path = st.text_input("Dataset CSV (optional for imputation)", "extracted_features_df.csv")
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
        st.info(f"Could not load dataset CSV (imputation will fall back to zeros): {e}")

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

            # Try prediction with smart fallbacks (string->numeric, then impute if NaN)
            with st.spinner("Scoring..."):
                proba, thresholds, note = attempt_predict_with_fallbacks(
                    model_artifact, X_aligned, dataset_df, thresholds_from_artifact, LABELS
                )

            if note:
                st.caption(note)

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

                        with st.spinner("Scoring..."):
                            proba, thresholds, note = attempt_predict_with_fallbacks(
                                model_artifact, X, dataset_df, thresholds_from_artifact, LABELS
                            )

                        if note:
                            st.caption(note)

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
                with st.spinner("Scoring..."):
                    proba, thresholds, note = attempt_predict_with_fallbacks(
                        model_artifact, X, dataset_df, thresholds_from_artifact, LABELS
                    )

                if note:
                    st.caption(note)

                proba_df = pd.DataFrame(proba, columns=LABELS)
                out_df = pd.concat([df_u.reset_index(drop=True), proba_df], axis=1)

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
        - **Gender**: `'M'` / `'F'`. If your model expects numeric, the app will fallback to `M→1, F→0`.
        - **Smoker**: `1` (Yes) / `0` (No).
        - **Audio (optional)**: The app computes many features your model likely expects:
          Zero Crossing Rate, RMS mean/std/skew, Spectral Centroid/Bandwidth/Contrast,
          Mel spectrogram mean/std, CQT mean/std/skew, Spectral Flux, MFCC means/std for first 4 bands,
          `mfcc_dev_*`, and `mfcc length`. OpenSMILE-specific fields remain NaN.

        **Imputation**
        - If your pipeline doesn't impute internally (e.g., Gradient Boosting), the app will impute:
          dataset column means when available, otherwise `0.0`. This enables prediction without audio.

        **Disclaimer**  
        Not a medical device. For educational/research use only.
        """
    )
