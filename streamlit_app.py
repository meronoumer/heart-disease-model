import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from typing import Any, Dict, Optional, Tuple, List

st.set_page_config(page_title="Heart Disease Ensemble (Stacked) Demo", page_icon="🫀", layout="centered")

# --------------------------------------------------------------------
# Settings / defaults
# --------------------------------------------------------------------
# Your merged model filename comes first:
CANDIDATE_MODEL_FILES = [
    "final_stacked_classifier_model.pkl",
    "hmm_multilabel_pipeline.joblib",  # fallback to older artifact
]

DEFAULT_LABELS = ["AS", "AR", "MR", "MS", "N"]
META_COLS = [
    "Unnamed: 0", "patient_id_x", "file_key", "audio_filename_base",
    "Age", "Gender", "Smoker", "Lives"
]
LABEL_COLS = DEFAULT_LABELS  # if labels exist in CSV; they won't be used at inference

# --------------------------------------------------------------------
# Loaders & helpers
# --------------------------------------------------------------------
def try_load_any(paths: List[str]) -> Tuple[Any, str]:
    last_err = None
    for p in paths:
        try:
            obj = load(p)
            return obj, p
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load any model file from {paths}. Last error: {last_err}")

@st.cache_resource
def load_artifact() -> Tuple[Any, str]:
    return try_load_any(CANDIDATE_MODEL_FILES)

@st.cache_data
def load_data(csv_path: str = "extracted_features_df.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)

def parse_mfcc_dev(s: str) -> list[float]:
    content = str(s).strip().lstrip("[").rstrip("]")
    if not content:
        return []
    try:
        return [float(x) for x in content.split()]
    except Exception:
        return []

def expand_mfcc_dev(df: pd.DataFrame) -> pd.DataFrame:
    if "mfcc devation" in df.columns:
        parsed = df["mfcc devation"].apply(parse_mfcc_dev)
        if len(parsed) and len(parsed.iloc[0]) > 0:
            n_feats = len(parsed.iloc[0])
            mfcc_cols = [f"mfcc_dev_{i}" for i in range(n_feats)]
            df_mfcc = pd.DataFrame(parsed.tolist(), columns=mfcc_cols, index=df.index)
            df = pd.concat([df.drop(columns=["mfcc devation"]), df_mfcc], axis=1)
    return df

def pick_feature_cols(df: pd.DataFrame, feature_cols_from_artifact: Optional[List[str]]) -> List[str]:
    if feature_cols_from_artifact:
        return feature_cols_from_artifact
    # Fallback: use all numeric columns except obvious meta/label columns
    exclude = set(META_COLS + LABEL_COLS)
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise RuntimeError("No numeric feature columns found to feed the model.")
    return num_cols

def sequence_indices(df: pd.DataFrame, file_key: Any) -> np.ndarray:
    return df.index[df["file_key"] == file_key].to_numpy()

def aggregate_probs(probs_2d: np.ndarray, how: str = "mean") -> np.ndarray:
    """
    probs_2d: shape (T, C) where T is frames, C is classes/labels.
    Returns shape (C,)
    """
    if probs_2d.ndim != 2:
        raise ValueError("Expected 2D probs array (frames x classes).")
    if how == "mean":
        return np.nanmean(probs_2d, axis=0)
    if how == "median":
        return np.nanmedian(probs_2d, axis=0)
    if how == "max":
        return np.nanmax(probs_2d, axis=0)
    return np.nanmean(probs_2d, axis=0)

def estimator_predict_proba_any(estimator, X: np.ndarray) -> np.ndarray:
    """
    Try to get per-sample probabilities for multi-label/multi-class and return (n_samples, n_labels).
    Handles common sklearn APIs (Pipeline, OneVsRest, Stacking, etc.).
    """
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        # Many estimators return (n_samples, n_classes).
        # Some multioutput classifiers return a list of arrays—handle that too.
        if isinstance(p, list):
            # List of (n_samples, 2) for each binary label -> stack [:, 1]
            cols = []
            for arr in p:
                arr = np.asarray(arr)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    cols.append(arr[:, 1])
                else:
                    cols.append(arr.ravel())
            return np.column_stack(cols)
        else:
            p = np.asarray(p)
            # If binary single-output with shape (n,2), return p[:,1:2]; if multi-label already (n,C), return as-is
            if p.ndim == 2 and p.shape[1] >= 2:
                return p if p.shape[1] > 2 else p[:, 1:2]
            return p.reshape(-1, 1)
    # Fallbacks
    if hasattr(estimator, "decision_function"):
        z = estimator.decision_function(X)
        z = np.asarray(z)
        # Sigmoid to map to (0,1)
        return 1.0 / (1.0 + np.exp(-z))
    if hasattr(estimator, "predict"):
        y = estimator.predict(X)
        return np.asarray(y).reshape(-1, 1).astype(float)
    # Last resort: 0.5
    return np.full((X.shape[0], 1), 0.5, dtype=float)

def hmm_delta(pos_hmm, neg_hmm, seq: np.ndarray) -> float:
    T = max(1, len(seq))
    return float((pos_hmm.score(seq) - neg_hmm.score(seq)) / T)

def hmm_to_prob(delta: float, center: float = 0.0, temp: float = 1.0) -> float:
    return float(1.0 / (1.0 + np.exp(-(delta - center) / max(1e-6, temp))))

# --------------------------------------------------------------------
# App
# --------------------------------------------------------------------
def main():
    st.title("🫀 Heart Disease Ensemble (Stacked) – Inference")
    st.write(
        "This app loads your **merged/stacked model** (`final_stacked_classifier_model.pkl`) "
        "and predicts per-label probabilities for a selected `file_key`."
    )

    # Load artifact
    try:
        artifact, fname = load_artifact()
    except Exception as e:
        st.error(f"Failed to load model artifact: {e}")
        st.stop()

    st.caption(f"Loaded: `{fname}`")

    # Load data
    try:
        df_raw = load_data()
    except Exception as e:
        st.error(f"Failed to read `extracted_features_df.csv`: {e}")
        st.stop()

    if "file_key" not in df_raw.columns:
        st.error("`file_key` column not found in CSV.")
        st.stop()

    # Expand MFCC deviation if present
    df = expand_mfcc_dev(df_raw.copy())

    # Branch 1: artifact is a dict (HMM + optional LR/RF path from earlier)
    if isinstance(artifact, dict):
        labels = artifact.get("labels", DEFAULT_LABELS)
        feature_cols = artifact.get("feature_cols", None)
        imputer = artifact.get("imputer", None)
        scaler = artifact.get("scaler", None)
        thresholds = artifact.get("thresholds", None)
        ensemble_thresholds = artifact.get("ensemble_thresholds", None)
        weights = artifact.get("ensemble_weights", None)

        # HMM path
        hmm_models = artifact.get("models") or artifact.get("hmm_models")
        lr_models = artifact.get("logreg_models")
        rf_models = artifact.get("rf_models")

        if any(v is None for v in [labels, imputer, scaler]):
            st.error("Artifact dict is missing one of: labels, imputer, scaler.")
            st.stop()

        # Prepare features
        if feature_cols is None:
            feature_cols = pick_feature_cols(df, None)
        X_raw = df[feature_cols].astype(float).values
        X_imp = imputer.transform(X_raw)
        X_scaled = scaler.transform(X_imp)

        st.write("**Detected components**:",
                 ", ".join([name for name, present in {
                     "HMM": hmm_models is not None,
                     "LogReg": lr_models is not None,
                     "RandomForest": rf_models is not None
                 }.items() if present]) or "none")

        # Pick file_key
        keys = sorted(df["file_key"].unique())
        fk = st.selectbox("Choose a file_key:", keys)
        idxs = sequence_indices(df, fk)
        if idxs.size == 0:
            st.warning("No rows found for that file_key.")
            st.stop()
        X_seq = X_scaled[idxs, :]

        # Score per label
        rows = []
        for lab in labels:
            out = {"label": lab, "prob_combined": None, "decision": None}
            parts: Dict[str, float] = {}

            # HMM
            if hmm_models and lab in hmm_models:
                pos_hmm, neg_hmm = hmm_models[lab]
                d = hmm_delta(pos_hmm, neg_hmm, X_seq)
                thr = thresholds.get(lab) if isinstance(thresholds, dict) else 0.0
                parts["hmm"] = hmm_to_prob(d, center=float(thr))

            # LR
            if lr_models and lab in lr_models:
                p = estimator_predict_proba_any(lr_models[lab], X_seq)
                parts["logreg"] = float(np.nanmean(p[:, -1]))  # positive class

            # RF
            if rf_models and lab in rf_models:
                p = estimator_predict_proba_any(rf_models[lab], X_seq)
                parts["rf"] = float(np.nanmean(p[:, -1]))

            if parts:
                # Weighted combine if weights provided (normalize), else equal
                keys_present = list(parts.keys())
                if weights:
                    w = np.array([max(0.0, float(weights.get(k, 0.0))) for k in keys_present], dtype=float)
                    if w.sum() == 0:
                        w = np.ones(len(keys_present))
                else:
                    w = np.ones(len(keys_present))
                w = w / w.sum()
                probs = np.array([parts[k] for k in keys_present], dtype=float)
                combined = float(np.dot(w, probs))
                out["prob_combined"] = combined

                # decide
                thr_final = None
                if isinstance(ensemble_thresholds, dict):
                    thr_final = ensemble_thresholds.get(lab, None)
                if thr_final is None:
                    thr_final = 0.5
                out["decision"] = int(combined >= thr_final)
            rows.append(out)

        st.subheader("Predictions (ensemble)")
        st.dataframe(pd.DataFrame(rows).set_index("label"))
        st.caption("Thresholds: ensemble_thresholds if provided; otherwise 0.5 on combined probability.")

        return  # end dict path

    # Branch 2: artifact is a fitted sklearn estimator (e.g., StackingClassifier or Pipeline)
    model = artifact
    st.write("**Detected artifact type:** sklearn estimator / pipeline")

    # If the model is a Pipeline that already includes preprocessing, you can feed raw features.
    # Otherwise we’ll pick numeric features (excluding meta/label columns).
    try:
        feature_cols = pick_feature_cols(df, None)
    except Exception as e:
        st.error(f"Could not determine feature columns: {e}")
        st.stop()

    # Choose a file_key
    keys = sorted(df["file_key"].unique())
    fk = st.selectbox("Choose a file_key:", keys)
    idxs = sequence_indices(df, fk)
    if idxs.size == 0:
        st.warning("No rows found for that file_key.")
        st.stop()

    # Build per-frame matrix for the chosen sequence
    X_seq = df.loc[idxs, feature_cols].astype(float).values

    # Get per-frame probabilities (n_frames, n_labels) then aggregate to sequence (n_labels,)
    probs_frames = estimator_predict_proba_any(model, X_seq)
    probs_seq = aggregate_probs(probs_frames, how="mean")  # mean across frames

    # Try to name labels. If the estimator exposes multi-class names, use them; else default.
    labels = DEFAULT_LABELS
    if hasattr(model, "classes_"):
        # If classes_ is (n_classes,) and matches what you trained, you can swap it here.
        # For multilabel wrappers, this may not map cleanly, so we keep DEFAULT_LABELS unless it's obvious.
        if isinstance(model.classes_, (list, np.ndarray)) and len(model.classes_) == len(probs_seq):
            labels = [str(c) for c in model.classes_]

    preds = (probs_seq >= 0.5).astype(int)
    out_df = pd.DataFrame({"probability": probs_seq, "predicted": preds}, index=labels)

    st.subheader("Predictions (stacked classifier)")
    st.dataframe(out_df)

    st.caption(
        "Notes:\n"
        "- We compute per-frame probabilities from your stacked classifier and **average** them to get a per-sequence prediction.\n"
        "- If your model already performs sequence aggregation internally (e.g., trained on sequence-level features), "
        "you can adapt the code to build those exact inputs instead of averaging frames. "
        "Share the expected input schema if you’d like me to wire that up precisely."
    )

if __name__ == "__main__":
    main()
