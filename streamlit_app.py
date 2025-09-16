from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Heart Disease Model – Predictor", layout="wide")


# =========================
# ------- Utilities -------
# =========================
def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric; invalids => NaN."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def impute_with_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    stats: dict[str, float],
    fallback: float = 0.0,
) -> pd.DataFrame:
    """
    Ensure df has exactly feature_cols in order; fill NaNs using stats then fallback.
    - Add missing expected columns (filled with stats[c] or fallback)
    - Drop unexpected columns
    - Coerce to numeric
    - Fill NaNs per column with stats, then fallback
    """
    df = df.copy()

    # Add missing expected columns
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = stats.get(c, fallback)

    # Drop extras
    extras = [c for c in df.columns if c not in feature_cols]
    if extras:
        df = df.drop(columns=extras)

    # Coerce
    df = coerce_numeric(df)

    # Fill NaNs
    for c in feature_cols:
        if c in stats:
            df[c] = df[c].fillna(stats[c])
        df[c] = df[c].fillna(fallback)

    # Order columns
    return df[feature_cols]


def prepare_features_for_inference(
    raw_features: pd.DataFrame,
    feature_cols: list[str],
    medians: dict[str, float] | None = None,
    means: dict[str, float] | None = None,
    prefer: str = "median",
) -> pd.DataFrame:
    """
    Choose which stats to use ('median' or 'mean') and impute -> aligned DataFrame.
    """
    if prefer not in {"median", "mean"}:
        prefer = "median"
    stats = medians if (prefer == "median" and medians) else (means or {})
    return impute_with_stats(raw_features, feature_cols, stats, fallback=0.0)


def _bundle_get(bundle: Any, *keys, default=None):
    """
    Safely fetch nested keys from common bundle formats:
      - dict at top-level
      - dict with 'bundle'/'artifacts'/'data'
      - simple objects with attributes
    Returns the first found among `keys`, else default.
    """
    # Dict-like
    if isinstance(bundle, dict):
        for k in keys:
            if k in bundle:
                return bundle[k]
        for container in ("bundle", "artifacts", "data"):
            if container in bundle and isinstance(bundle[container], dict):
                for k in keys:
                    if k in bundle[container]:
                        return bundle[container][k]
    # Attribute-style
    for k in keys:
        if hasattr(bundle, k):
            return getattr(bundle, k)
    return default


@st.cache_resource
def load_artifacts_from_bundle(
    bundle_path: str | Path = "artifacts/final_stacked_model_bundle.pkl",
):
    """
    Load pipeline/model + feature list + stats from a single bundle.

    Returns dict:
      - predict_fn: callable(X) -> y
      - proba_fn: callable(X) -> probabilities or None
      - needs_manual_impute: bool
      - feature_cols: list[str]
      - medians: dict[str, float]
      - means: dict[str, float]
    """
    bundle = joblib.load(bundle_path)

    # model / pipeline
    pipeline_or_model = _bundle_get(bundle, "pipeline", "model", "final_model", "estimator")
    if pipeline_or_model is None or not hasattr(pipeline_or_model, "predict"):
        raise ValueError(
            "No predict-capable object found in bundle (expected keys like 'pipeline'/'model')."
        )

    # feature columns
    feature_cols = _bundle_get(bundle, "feature_cols", "features", "feature_columns")
    if feature_cols is None:
        meta = _bundle_get(bundle, "metadata", "config", default={}) or {}
        feature_cols = meta.get("feature_cols")
    if feature_cols is None:
        stats_container = _bundle_get(bundle, "stats", "statistics", default={}) or {}
        feature_cols = stats_container.get("feature_cols")

    if feature_cols is None:
        raise ValueError("feature_cols not found in bundle. Include exact training order.")

    # stats
    medians = _bundle_get(bundle, "medians", "feature_medians") or {}
    means = _bundle_get(bundle, "means", "feature_means") or {}
    stats_container = _bundle_get(bundle, "stats", "statistics") or {}
    if isinstance(stats_container, dict):
        medians = stats_container.get("medians", medians) or medians
        means = stats_container.get("means", means) or means

    # decide if we must impute here (safe default: True if stats exist)
    needs_manual_impute = True

    # if it's clearly a Pipeline with an imputer, we can skip manual impute
    step_names = []
    if hasattr(pipeline_or_model, "steps"):
        step_names = [name for (name, _) in pipeline_or_model.steps]
    elif hasattr(pipeline_or_model, "named_steps"):
        step_names = list(pipeline_or_model.named_steps.keys())
    imputerish = {"imputer", "simpleimputer", "preprocess", "preprocessor", "columntransformer"}
    if any(any(tok in name.lower() for tok in imputerish) for name in step_names):
        needs_manual_impute = False

    # hist gradient boosting natively handles NaNs
    if "histgradientboosting" in pipeline_or_model.__class__.__name__.lower():
        needs_manual_impute = False

    # If bundle includes explicit medians/means, prefer manual impute for safety.
    if medians or means:
        needs_manual_impute = True

    # predict/proba functions
    def predict_fn(X: pd.DataFrame):
        return pipeline_or_model.predict(X)

    proba_fn = getattr(pipeline_or_model, "predict_proba", None)
    if not callable(proba_fn):
        proba_fn = None

    return {
        "predict_fn": predict_fn,
        "proba_fn": proba_fn,
        "needs_manual_impute": needs_manual_impute,
        "feature_cols": list(feature_cols),
        "medians": medians,
        "means": means,
        "pipeline_or_model": pipeline_or_model,
    }


# =========================
# --------- App ----------
# =========================
def main():
    st.title("Cardiac Murmur / Disease Classifier (BMD-HS)")
    st.caption("Robust inference with median-based imputation & strict feature alignment")

    # ---------- Load bundle ----------
    try:
        art = load_artifacts_from_bundle("artifacts/final_stacked_model_bundle.pkl")
    except Exception as e:
        st.error(f"Could not load model bundle: {e}")
        st.stop()

    predict_fn = art["predict_fn"]
    proba_fn = art["proba_fn"]
    needs_manual_impute = art["needs_manual_impute"]
    feature_cols = art["feature_cols"]
    medians = art["medians"]
    means = art["means"]

    # ---------- Input mode ----------
    tab_single, tab_batch = st.tabs(["Single sample (JSON/dict)", "Batch CSV"])

    with tab_single:
        st.write("Paste a **JSON object** with your feature values (keys = feature names).")
        default_example = {c: 0 for c in feature_cols[:10]}  # short example with 10 keys
        json_text = st.text_area(
            "Feature JSON",
            value=json.dumps(default_example, indent=2),
            height=220,
        )
        colA, colB = st.columns([1, 2])
        with colA:
            prefer = st.radio("Impute with", options=["median", "mean"], index=0, horizontal=True)
        with colB:
            run_single = st.button("Predict (single)", use_container_width=True)

        if run_single:
            try:
                payload = json.loads(json_text)
                raw_df = pd.DataFrame([payload])
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                st.stop()

            if needs_manual_impute:
                X = prepare_features_for_inference(
                    raw_df, feature_cols, medians, means, prefer=prefer
                )
            else:
                # pipeline handles it; still coerce to numeric
                X = coerce_numeric(raw_df).reindex(columns=feature_cols, fill_value=np.nan)

            show_debug(raw_df, X, feature_cols, needs_manual_impute)

            try:
                y_pred = predict_fn(X)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            st.subheader("Prediction")
            st.write(np.asarray(y_pred).tolist())

            if proba_fn is not None:
                try:
                    probs = proba_fn(X)
                    st.subheader("Probabilities")
                    st.write(np.asarray(probs).tolist())
                except Exception:
                    pass

    with tab_batch:
        st.write("Upload a **CSV** where columns are your feature names.")
        uploaded = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)
        prefer2 = st.radio("Impute with", options=["median", "mean"], index=0, horizontal=True, key="impute_batch")
        run_batch = st.button("Predict (batch)", use_container_width=True, key="run_batch")

        if run_batch:
            if not uploaded:
                st.error("Please upload a CSV first.")
                st.stop()
            try:
                raw_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                st.stop()

            if needs_manual_impute:
                X = prepare_features_for_inference(
                    raw_df, feature_cols, medians, means, prefer=prefer2
                )
            else:
                X = coerce_numeric(raw_df).reindex(columns=feature_cols, fill_value=np.nan)

            show_debug(raw_df, X, feature_cols, needs_manual_impute)

            try:
                y_pred = predict_fn(X)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            st.subheader("Predictions")
            st.dataframe(pd.DataFrame({"prediction": np.asarray(y_pred).tolist()}))

            if proba_fn is not None:
                try:
                    probs = proba_fn(X)
                    if isinstance(probs, list) or (isinstance(probs, np.ndarray) and probs.ndim == 1):
                        # binary or single-task
                        st.subheader("Probabilities")
                        st.dataframe(pd.DataFrame({"prob": np.asarray(probs).tolist()}))
                    else:
                        st.subheader("Probabilities (per class)")
                        st.dataframe(pd.DataFrame(np.asarray(probs)))
                except Exception:
                    pass

    with st.expander("About"):
        st.markdown(
            """
            - **NaN-safe inference**: inputs are coerced to numeric; missing values are imputed with **saved medians** (or means).
            - **Strict column alignment**: we enforce the exact training **feature order** to avoid silent drift.
            - If your bundle contains a Pipeline with an imputer, we can bypass manual impute (auto-detected).
            """
        )


def show_debug(raw_df: pd.DataFrame, X: pd.DataFrame, feature_cols: list[str], needs_manual_impute: bool):
    missing_cols = [c for c in feature_cols if c not in raw_df.columns]
    unexpected_cols = [c for c in raw_df.columns if c not in feature_cols]
    with st.expander("Debug: Feature alignment and preprocessing", expanded=False):
        st.write("needs_manual_impute:", needs_manual_impute)
        st.write("Expected feature count:", len(feature_cols))
        st.write("Incoming columns (sample):", list(raw_df.columns)[:25])
        if missing_cols:
            st.info(f"Imputed {len(missing_cols)} missing feature(s): {missing_cols[:12]}{' ...' if len(missing_cols) > 12 else ''}")
        if unexpected_cols:
            st.warning(f"Dropped {len(unexpected_cols)} unexpected feature(s): {unexpected_cols[:12]}{' ...' if len(unexpected_cols) > 12 else ''}")
        st.write("Model input shape:", X.shape)
        st.write("Any NaNs remaining in X?:", bool(pd.isna(X).any().any()))


if __name__ == "__main__":
    main()
