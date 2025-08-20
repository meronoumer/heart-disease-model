# streamlit_app.py
# ------------------------------------------------------------
# Heart Disease Model Demo App
#
# Modes:
# 1) Manual Entry (Age/Gender/Smoker/Lives + optional Audio -> MFCC features)
# 2) Pick from dataset (extracted_features_df.csv)
# 3) Upload CSV (single/batch)
#
# Model path, dataset path, and labels are configurable via the sidebar.
# The app aligns inputs to the model's expected feature columns if present
# in the artifact (e.g., "feature_cols"). Missing columns are set to NaN
# so your pipeline can impute.
#
# ------------------------------------------------------------

from __future__ import annotations

import io
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Optional audio dependencies
try:
    import librosa
except Exception:
    librosa = None

# Reduce noisy HMM warnings if present in artifact
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

# -------------------------- Utilities --------------------------
def parse_mfcc_dev_str(s: str) -> List[float]:
    """Parse a string like '[0.1 0.2 0.3]' into a list of floats."""
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
        return df.drop(columns=[col])
    n_feats = len(parsed.iloc[0])
    mfcc_cols = [f"mfcc_dev_{i}" for i in range(n_feats)]
    df_mfcc = pd.DataFrame(parsed.tolist(), columns=mfcc_cols, index=df.index)
    return pd.concat([df.drop(columns=[col]), df_mfcc], axis=1)

def align_features(X: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Align X to required feature_cols; create missing as NaN; drop extras."""
    missing = [c for c in feature_cols if c not in X.columns]
    extra = [c for c in X.columns if c not in feature_cols]
    for c in missing:
        X[c] = np.nan
    X = X[feature_cols]
    return X, missing, extra

def guess_feature_cols_from_artifact(model_art):
    """Try to read feature columns from the artifact."""
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
    """
    Produce (n_samples, n_labels) probabilities.
    Supports:
    - dict bundles with callable 'predict_proba' or nested estimators
    - scikit estimators/pipelines with predict_proba
    """
    # dict bundle patterns
    if isinstance(model_art, dict):
        if "predict_proba" in model_art and callable(model_art["predict_proba"]):
            return np.asarray(model_art["predict_proba"](X))
        for key in ("pipeline", "model", "clf", "stacked"):
            est = model_art.get(key)
            if est is not None and hasattr(est, "predict_proba"):
                proba = est.predict_proba(X)
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
    # bare estimator/pipeline
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

    raise RuntimeError("Loaded artifact does not expose predict_proba.")

def aggregate_group_rows_to_single_vector(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Convert many rows per file_key into a single feature row.
    Expands 'mfcc devation' if present and averages numeric columns.
    """
    g = expand_mfcc_dev_column(grp.copy(), col="mfcc devation")
    cols_to_skip = set(DEFAULT_META_COLS + DEFAULT_LABEL_COLS)
    num_cols = [c for c in g.columns if c not in cols_to_skip and pd.api.types.is_numeric_dtype(g[c])]
    if not num_cols:
        return g.iloc[[0]].drop(columns=[c for c in g.columns if c in DEFAULT_LABEL_COLS], errors="ignore")
    return g[num_cols].mean(axis=0, skipna=True).to_frame().T

def human_prob(p: float) -> str:
    return f"{100.0 * float(p):.1f}%"

# -------------------------- Caching --------------------------
@st.cache_resource(show_spinner=False)
def load_model_resource(path: str):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# -------------------------- Sidebar --------------------------
st.title("🫀 Heart Disease Model — Interactive Demo")
st.markdown(
    "Manipulate **patient inputs** like Age, Gender, Smoker, and optionally upload **audio** to "
    "generate MFCC features for prediction with your **merged ensemble model**.\n\n"
    "**Disclaimer:** Educational/research use only — not a medical device."
)

with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input(
        "Model file path",
        value="final_stacked_classifier_model.pkl",
        help="Path to your merged model artifact.",
    )
    dataset_path = st.text_input(
        "Dataset CSV (optional)",
        value="extracted_features_df.csv",
        help="Used in 'Pick from dataset' and for reference.",
    )
    labels_input = st.text_input(
        "Label names (comma-separated)",
        value="AS,AR,MR,MS,N",
        help="Used for display if the artifact doesn't define labels.",
    )
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

# -------------------------- Tabs --------------------------
tab_manual, tab_dataset, tab_csv, tab_about = st.tabs(
    ["🧑‍⚕️ Manual Entry", "📁 Pick from Dataset", "📦 Upload CSV", "ℹ️ About"]
)

# -------------------------- Manual Entry --------------------------
with tab_manual:
    st.subheader("Patient Inputs (manual) + Optional Audio")

    # Patient inputs
    colA, colB, colC, colD = st.columns(4)
    with colA:
        age = st.number_input("Age", min_value=0, max_value=120, value=60, step=1)
    with colB:
        gender = st.selectbox("Gender", options=["Male", "Female", "Other", "Unknown"], index=0)
    with colC:
        smoker = st.selectbox("Smoker", options=["No", "Yes", "Unknown"], index=0)
    with colD:
        lives = st.selectbox("Lives (region)", options=["Urban", "Suburban", "Rural", "Unknown"], index=0)

    # Optional audio upload -> create mfcc_dev_* features
    st.markdown("**Optional:** Upload an audio file to extract MFCC deviation features.")
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

    # Optional: show/override any additional numeric features the model expects
    with st.expander("Advanced: Add/override extra numeric features (optional)"):
        st.caption("If your model expects more inputs (e.g., engineered features), you can supply them here.")
        extra_kv = st.data_editor(
            pd.DataFrame({"feature": [], "value": []}),
            num_rows="dynamic",
            use_container_width=True,
            key="extra_kv_editor"
        )

    def make_manual_row_df() -> pd.DataFrame:
        """Build a single-row DataFrame from manual widgets + optional audio MFCCs."""
        row = {
            "Age": age,
            "Gender": gender,
            "Smoker": smoker,
            "Lives": lives,
        }

        # Add extra numeric features from editor
        if isinstance(extra_kv, pd.DataFrame) and {"feature", "value"}.issubset(extra_kv.columns):
            for _, r in extra_kv.iterrows():
                f = str(r.get("feature", "")).strip()
                if f:
                    try:
                        row[f] = float(r.get("value"))
                    except Exception:
                        # Keep raw string if not convertible
                        row[f] = r.get("value")

        # Add MFCC deviation features if audio provided
        if librosa is not None and audio_file is not None:
            try:
                y, sr = librosa.load(io.BytesIO(audio_file.read()), sr=sr_target, mono=True)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)   # (n_mfcc, n_frames)
                dev = np.std(mfcc, axis=1)                               # deviation across frames
                for i in range(n_mfcc):
                    row[f"mfcc_dev_{i}"] = float(dev[i])
            except Exception as e:
                st.error(f"Audio processing failed: {e}")

        return pd.DataFrame([row])

    if st.button("Run prediction", type="primary"):
        try:
            row_df = make_manual_row_df()

            # If dataset-like "mfcc devation" text were present, expand (unlikely in manual)
            row_df = expand_mfcc_dev_column(row_df, col="mfcc devation")

            # Determine feature columns to align to
            feat_cols = feature_cols_from_artifact
            if feat_cols is None:
                # Best guess: use all columns except labels if any
                feat_cols = [c for c in row_df.columns if c not in DEFAULT_LABEL_COLS]

            X_aligned, missing, extra = align_features(row_df.copy(), feat_cols)
            if missing:
                st.info(f"Missing features were set to NaN (pipeline should impute): {missing}")
            if extra:
                st.caption(f"Ignored extra columns: {extra}")

            with st.spinner("Scoring..."):
                proba = predict_proba_from_artifact(model_artifact, X_aligned)
                proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)

            # Decisions use thresholds from artifact if available, else 0.5
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
                        row_df = aggregate_group_rows_to_single_vector(sub)

                        # Expand if string-based MFCC exists
                        row_df = expand_mfcc_dev_column(row_df, col="mfcc devation")

                        feat_cols = feature_cols_from_artifact
                        if feat_cols is None:
                            # Derive by removing obvious label columns from this aggregate row
                            feat_cols = [c for c in row_df.columns if c not in DEFAULT_LABEL_COLS]

                        X, missing, extra = align_features(row_df.copy(), feat_cols)
                        if missing:
                            st.info(f"Missing features set to NaN: {missing}")
                        if extra:
                            st.caption(f"Ignored extras: {extra}")

                        with st.spinner("Scoring..."):
                            proba = predict_proba_from_artifact(model_artifact, X)
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
                    row = aggregate_group_rows_to_single_vector(grp)
                    row[DEFAULT_SEQ_KEY] = str(key)
                    outs.append(row)
                df_u = pd.concat(outs, ignore_index=True)

            feat_cols = feature_cols_from_artifact
            if feat_cols is None:
                feat_cols = [c for c in df_u.columns if c not in DEFAULT_LABEL_COLS]

            X, missing, extra = align_features(df_u.copy(), feat_cols)
            if missing:
                st.info(f"Missing features created as NaN: {missing}")
            if extra:
                st.caption(f"Ignored extras: {extra}")

            if st.button("Run CSV prediction", type="primary"):
                with st.spinner("Scoring..."):
                    proba = predict_proba_from_artifact(model_artifact, X)
                    proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)

                proba_df = pd.DataFrame(proba, columns=LABELS)
                out_df = pd.concat([df_u.reset_index(drop=True), proba_df], axis=1)

                thresholds = thresholds_from_artifact or {lab: 0.5 for lab in LABELS}
                for lab in LABELS:
                    thr = float(thresholds.get(lab, 0.5))
                    out_df[f"{lab}_pred"] = (out_df[lab] >= thr).astype(int)

                st.success("Batch prediction complete.")
                st.dataframe(out_df.head(100), use_container_width=True)

                # Download
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
        **Purpose**  
        This app demonstrates a merged heart disease classifier. Users can adjust patient
        inputs (Age, Gender, Smoker, Lives) and optionally upload audio that is converted to
        MFCC deviation features.

        **How inputs are handled**  
        - The app tries to align inputs to the model’s expected `feature_cols` found in the
          artifact. Any missing features are set to NaN (your preprocessing should impute).
        - Categorical fields (Gender/Smoker/Lives) are passed as strings; your pipeline should
          include encoders (e.g., OneHotEncoder). If your model expects numeric encodings,
          adjust the manual fields or add mappings in the Advanced section.

        **Disclaimer**  
        Not a medical device. For educational and research use only.
        """
    )
