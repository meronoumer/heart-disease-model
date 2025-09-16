from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, List, Dict

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================
#  Predict-safety shim: add NaN-proof inference to any UI
#  Usage in your app:
#     from predict_safe_shim import predict_safe, load_artifacts_from_bundle
#     y_pred, probs, X_infer, feature_cols, needs_manual = predict_safe(raw_df, prefer="median")
# =============================================================

# ----------------------- Helpers ----------------------------

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric; invalids become NaN."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def impute_with_stats(
    df: pd.DataFrame,
    feature_cols: List[str],
    stats: Dict[str, float],
    fallback: float = 0.0,
) -> pd.DataFrame:
    """Ensure df has exactly feature_cols in order; fill NaNs using stats then fallback."""
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
    feature_cols: List[str],
    medians: Dict[str, float] | None = None,
    means: Dict[str, float] | None = None,
    prefer: str = "median",
) -> pd.DataFrame:
    """Choose which stats to use ('median' or 'mean') and impute -> aligned DataFrame."""
    if prefer not in {"median", "mean"}:
        prefer = "median"
    stats = medians if (prefer == "median" and medians) else (means or {})
    return impute_with_stats(raw_features, feature_cols, stats, fallback=0.0)


# -------------------- Bundle loading ------------------------

def _bundle_get(bundle: Any, *keys, default=None):
    """Safely fetch nested keys from common bundle formats."""
    # Dict-like
    if isinstance(bundle, dict):
        for k in keys:
            if k in bundle:
                return bundle[k]
        for container in ("bundle", "artifacts", "data", "stats", "statistics", "metadata", "config"):
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
      - model_or_pipeline: the predict-capable object
      - predict_fn: callable(X) -> y
      - proba_fn: callable(X) -> probabilities or None
      - needs_manual_impute: bool
      - feature_cols: list[str]
      - medians: dict[str, float]
      - means: dict[str, float]
    """
    bundle = joblib.load(bundle_path)

    model_or_pipeline = _bundle_get(bundle, "pipeline", "model", "final_model", "estimator")
    if model_or_pipeline is None or not hasattr(model_or_pipeline, "predict"):
        raise ValueError("No predict-capable object found in bundle (keys like 'pipeline'/'model').")

    feature_cols = _bundle_get(bundle, "feature_cols", "features", "feature_columns")
    if feature_cols is None:
        meta = _bundle_get(bundle, "metadata", "config", default={}) or {}
        feature_cols = meta.get("feature_cols")
    if feature_cols is None:
        stats_container = _bundle_get(bundle, "stats", "statistics", default={}) or {}
        feature_cols = stats_container.get("feature_cols")
    if feature_cols is None:
        raise ValueError("feature_cols not found in bundle. Include exact training order.")

    medians = _bundle_get(bundle, "medians", "feature_medians") or {}
    means = _bundle_get(bundle, "means", "feature_means") or {}
    stats_container = _bundle_get(bundle, "stats", "statistics") or {}
    if isinstance(stats_container, dict):
        medians = stats_container.get("medians", medians) or medians
        means = stats_container.get("means", means) or means

    # Decide if we should handle NaNs here (safe default: True if stats are present)
    needs_manual_impute = True
    step_names = []
    if hasattr(model_or_pipeline, "steps"):
        step_names = [name for (name, _) in model_or_pipeline.steps]
    elif hasattr(model_or_pipeline, "named_steps"):
        step_names = list(model_or_pipeline.named_steps.keys())
    imputerish = {"imputer", "simpleimputer", "preprocess", "preprocessor", "columntransformer"}
    if any(any(tok in name.lower() for tok in imputerish) for name in step_names):
        needs_manual_impute = False
    if "histgradientboosting" in model_or_pipeline.__class__.__name__.lower():
        needs_manual_impute = False
    if medians or means:
        needs_manual_impute = True

    def predict_fn(X: pd.DataFrame):
        return model_or_pipeline.predict(X)

    proba_fn = getattr(model_or_pipeline, "predict_proba", None)
    if not callable(proba_fn):
        proba_fn = None

    return {
        "model_or_pipeline": model_or_pipeline,
        "predict_fn": predict_fn,
        "proba_fn": proba_fn,
        "needs_manual_impute": needs_manual_impute,
        "feature_cols": list(feature_cols),
        "medians": medians,
        "means": means,
    }


# ------------------ One-call safe predict -------------------

def predict_safe(
    raw_features_df: pd.DataFrame,
    prefer: str = "median",
    bundle_path: str | Path = "artifacts/final_stacked_model_bundle.pkl",
) -> Tuple[np.ndarray, np.ndarray | None, pd.DataFrame, List[str], bool]:
    """
    Returns: (y_pred, probs_or_None, X_infer, feature_cols, needs_manual_impute)
    """
    art = load_artifacts_from_bundle(bundle_path)
    feature_cols = art["feature_cols"]

    if art["needs_manual_impute"]:
        X = prepare_features_for_inference(
            raw_features_df, feature_cols, art["medians"], art["means"], prefer=prefer
        )
    else:
        # Let the pipeline handle it; still coerce to numeric and align columns
        X = coerce_numeric(raw_features_df).reindex(columns=feature_cols, fill_value=np.nan)

    if X.isna().any().any():
        # Defensive: should not happen after manual impute, but avoid model crashes
        X = X.fillna(0.0)

    y_pred = art["predict_fn"](X)
    probs = None
    if art["proba_fn"] is not None:
        try:
            probs = art["proba_fn"](X)
        except Exception:
            probs = None

    return y_pred, probs, X, feature_cols, art["needs_manual_impute"]
