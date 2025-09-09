# streamlit_app.py
import os
import io
import warnings
import pandas as pd
import streamlit as st
import joblib
import io
try:
    import soundfile as sf  # audio reader for wav/flac/ogg
except Exception:
    sf = None
import numpy as np  # if not already imported


st.set_page_config(page_title="Cardiovascular Diagnostic Aid", page_icon="🫀", layout="centered")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PRIMARY_LABELS = ["AS", "AR", "MR", "MS", "N"]

def _safe_load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    try:
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

@st.cache_resource(show_spinner=True)
def load_model_bundle(model_path: str):
    bundle = _safe_load(model_path)
    pipe, feature_cols, labels = None, None, None

    if isinstance(bundle, dict):
        pipe = bundle.get("pipeline") or bundle.get("model") or bundle.get("estimator") or bundle
        feature_cols = bundle.get("feature_cols")
        labels = bundle.get("labels") or bundle.get("classes") or bundle.get("class_names")
    else:
        pipe = bundle

    if feature_cols is None:
        if hasattr(pipe, "feature_names_in_"):
            feature_cols = list(pipe.feature_names_in_)
        else:
            try:
                if hasattr(pipe, "named_steps"):
                    for _, step in pipe.named_steps.items():
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
        feature_cols = []

    if labels is None:
        if hasattr(pipe, "classes_"):
            cls = list(getattr(pipe, "classes_"))
            if len(cls) and isinstance(cls[0], (list, np.ndarray)):
                try:
                    cls = [str(c[1]) if len(c) > 1 else str(c[0]) for c in cls]
                except Exception:
                    cls = PRIMARY_LABELS
            else:
                cls = [str(c) for c in cls]
            labels = cls
        else:
            labels = PRIMARY_LABELS

    return pipe, feature_cols, labels

def ensure_feature_schema(row_dict: dict, feature_cols: list[str]) -> pd.DataFrame:
    if not feature_cols:
        X = pd.DataFrame([row_dict])
    else:
        X = pd.DataFrame([{c: row_dict.get(c, np.nan) for c in feature_cols}])
    # Coerce any object dtype to numeric
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def show_missing_features(missing_cols: list[str], reason: str, *, compact_if_audio_only=True):
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

def compute_audio_features(file: io.BytesIO) -> dict:
    # Placeholder; relying on imputation unless you want me to wire librosa later.
    return {}

def _bars_from_probs(probs_df: pd.DataFrame):
    for col in probs_df.columns:
        st.write(f"**{col}**")
        st.progress(float(probs_df.iloc[0][col]))

# NEW: try/auto-encode Gender (M/F) -> (1/0) only if the model complains about strings.
def _encode_gender_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    mapping = {"M": 1, "F": 0, "Male": 1, "Female": 0, "m": 1, "f": 0}
    X2["Gender"] = X2["Gender"].map(mapping).astype(float)
    return X2

def _predict_with_gender_fallback(model, X: pd.DataFrame):
    """
    Try predict_proba / decision_function / predict.
    If we hit a 'string to float' error and Gender is object, retry with Gender encoded to 0/1.
    Returns (np.ndarray probs_like, DataFrame X_used)
    """
    def _as_probs(y_like):
        y = np.array(y_like)
        if isinstance(y_like, list):
            y = np.array([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in y_like]).T
        return np.atleast_2d(y)

    # attempt 1: as-is
    try:
        return _as_probs(model.predict_proba(X)), X
    except Exception as e1:
        msg = str(e1)

    # attempt 1b: decision_function as-is
    try:
        scores = np.atleast_2d(model.decision_function(X))
        return 1 / (1 + np.exp(-scores)), X
    except Exception as e2:
        msg = f"{msg} | {e2}"

    # if error suggests string->float and Gender is object, retry with numeric encoding
    if "could not convert string to float" in msg or "dtype='O'" in msg or "convert" in msg:
        if "Gender" in X.columns and X["Gender"].dtype == object:
            X_num = _encode_gender_numeric(X)

            # retry with proba
            try:
                return _as_probs(model.predict_proba(X_num)), X_num
            except Exception:
                pass
            # retry with decision_function
            try:
                scores = np.atleast_2d(model.decision_function(X_num))
                return 1 / (1 + np.exp(-scores)), X_num
            except Exception:
                pass
            # last resort: predict -> pseudo-prob
            preds = np.atleast_1d(model.predict(X_num))
            return np.atleast_2d(preds.astype(float)), X_num

    # generic last resort on original X
    try:
        preds = np.atleast_1d(model.predict(X))
        return np.atleast_2d(preds.astype(float)), X
    except Exception as e3:
        raise e3  # bubble up to show the real root cause
    
def render_audio_uploader():
    st.subheader("Optional: Upload a heart/voice audio sample")
    st.caption("Supported: WAV, FLAC, OGG. (MP3 needs extra setup; see note below.)")

    uploaded = st.file_uploader(
        "Choose an audio file",
        type=["wav", "flac", "ogg"],
        key="audio_file",
        help="This audio is for reference only and is not used by the prediction model."
    )
    if not uploaded:
        return None

    # Keep the raw bytes for the built-in player
    file_bytes = uploaded.getvalue()
    st.audio(file_bytes, format=uploaded.type or "audio/wav")

    if sf is None:
        st.warning("Audio analysis is disabled because the 'soundfile' package is unavailable.")
        return None

    # Basic, dependency-light audio summary
    try:
        data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
        # Mono-ize if needed
        if data.ndim > 1:
            data = data.mean(axis=1)

        duration = float(len(data) / sr) if sr else 0.0
        rms = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
        # Simple zero-crossing rate without librosa
        zc = int(((data[:-1] * data[1:]) < 0).sum()) if len(data) > 1 else 0
        zcr = float(zc / (len(data) - 1)) if len(data) > 1 else 0.0

        st.caption(f"Sample rate: {sr} Hz • Duration: {duration:.2f}s • RMS: {rms:.4f} • ZCR: {zcr:.4f}")

        # Return a small dict you could log or save if you want (not used in prediction)
        return {"audio_duration_s": duration, "audio_rms": rms, "audio_zcr": zcr}
    except Exception as e:
        st.warning(f"Could not parse audio: {e}")
        return None


# ===========================
# App UI
# ===========================
st.title("🫀 Cardiovascular Disease Diagnostic Aid")
st.markdown(
    "Upload patient attributes and (optionally) an audio file. "
    "This app uses your trained model to estimate label probabilities."
)

with st.sidebar:
    st.header("Settings")
    default_model_path = "models/final_stacked_classifier_model.pkl"
    model_path = st.text_input("Model file path", value=default_model_path)
    show_expected_cols = st.checkbox("Show expected feature columns after loading", value=False)

st.header("Patient Information")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
with c2:
    # Show M/F to the user, send 1.0/0.0 to the model
    gender_choice = st.selectbox(
        "Gender",
        options=[("M", 1.0), ("F", 0.0)],
        index=0,
        format_func=lambda opt: opt[0],  # display only M or F
        help="Displayed as M/F, encoded to 1/0 for the model."
    )
    gender = gender_choice[1]
with c3:
    smoker = st.selectbox(
        "Smoker",
        options=[0, 1],
        index=0,
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="0 = No, 1 = Yes"
    )

st.header("Optional: Upload Audio")
audio_file = st.file_uploader("Upload heart sound/audio file", type=["wav", "mp3", "flac", "ogg"])

st.divider()
predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

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

        row = {
            "Age": age,
            "Gender": float(gender),      # may be string ('M'/'F'); we will auto-encode if the model needs numbers
            "Smoker": float(smoker), # 0/1
        }

        if audio_file is not None:
            try:
                row.update(compute_audio_features(audio_file))
                audio_reason = "audio uploaded, but some features may still be imputed"
            except Exception as e:
                st.warning(f"Audio feature extraction failed; proceeding with imputation. ({e})")
                audio_reason = "audio uploaded but features not extracted"
        else:
            audio_reason = "no audio uploaded"

        X = ensure_feature_schema(row, feature_cols)
        missing = [c for c in feature_cols if c in X.columns and pd.isna(X.at[0, c])]
        show_missing_features(missing, audio_reason)

        # 🔧 Robust prediction (auto-encode Gender if required)
        proba_like, X_used = _predict_with_gender_fallback(model, X)

        # Harmonize label names vs. proba shape
        label_names = labels or PRIMARY_LABELS
        if proba_like.ndim == 2 and proba_like.shape[1] != len(label_names):
            label_names = [f"class_{i}" for i in range(proba_like.shape[1])]

        probs_df = pd.DataFrame(proba_like, columns=label_names).clip(0.0, 1.0)

        st.subheader("Predicted Probabilities")
        _bars_from_probs(probs_df)

        with st.expander("Show raw prediction table"):
            st.dataframe(probs_df.style.format("{:.3f}"), use_container_width=True)

        pred_labels = [lbl for lbl, p in zip(probs_df.columns, probs_df.iloc[0].values) if p >= 0.5]
        st.success(f"Predicted label(s): {', '.join(pred_labels) if pred_labels else 'None ≥ 0.5'}")

        # Small hint when we had to encode Gender
        if "Gender" in X.columns and X["Gender"].dtype == object and "Gender" in X_used.columns and X_used["Gender"].dtype != object:
            st.caption("Note: Gender was auto-encoded to numeric (M→1, F→0) for this model.")

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
