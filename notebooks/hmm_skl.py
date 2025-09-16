# hmm_skl.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Union, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from hmmlearn.hmm import GaussianHMM
from scipy.special import expit

def _stack_sequences(
    X: Union[List[np.ndarray], Tuple[np.ndarray, List[int]], np.ndarray]
) -> Tuple[np.ndarray, List[int], List[np.ndarray]]:
    # Accept: list[(Ti,D)], (X_concat, lengths), or 3D array (n,T,D)
    if isinstance(X, tuple) and len(X) == 2:
        Xc, lengths = X
        seqs, s = [], 0
        for L in lengths:
            seqs.append(Xc[s:s+L]); s += L
        return np.asarray(Xc), list(map(int, lengths)), seqs
    if isinstance(X, list):
        lengths = [xi.shape[0] for xi in X]
        Xc = np.vstack(X)
        return Xc, lengths, X
    X = np.asarray(X)
    if X.ndim == 3:                              # (n, T, D)
        n, T, D = X.shape
        return X.reshape(n*T, D), [T]*n, [X[i] for i in range(n)]
    raise ValueError("X must be list[(Ti,D)], (X_concat,lengths), or (n,T,D)")

class HMMBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible binary classifier built on hmmlearn.GaussianHMM.
    Trains one HMM for class 0 and one for class 1; uses log-likelihood ratio.
    """
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        n_iter: int = 100,
        tol: float = 1e-2,
        random_state: Optional[int] = None,
        verbose: bool = False,
        scale: bool = True,           # standardize per-frame features
        prior_smoothing: float = 1.0, # Laplace prior smoothing
        normalize_ll: bool = True,    # divide log-likelihood by T
        threshold: float = 0.5        # decision threshold for predict()
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.scale = scale
        self.prior_smoothing = prior_smoothing
        self.normalize_ll = normalize_ll
        self.threshold = threshold

    # ---- sklearn plumbing ----
    def get_params(self, deep: bool = True):
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "scale": self.scale,
            "prior_smoothing": self.prior_smoothing,
            "normalize_ll": self.normalize_ll,
            "threshold": self.threshold,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ---- internal helpers ----
    def _smooth_probs(self, hmm, eps=1e-6):
        hmm.transmat_  = (np.nan_to_num(hmm.transmat_,  nan=0.0) + eps)
        hmm.transmat_  /= hmm.transmat_.sum(axis=1, keepdims=True)
        hmm.startprob_ = (np.nan_to_num(hmm.startprob_, nan=0.0) + eps)
        hmm.startprob_ /= hmm.startprob_.sum()

    def _fit_hmm(self, seqs, rs):
        Xc = np.vstack(seqs)
        Ls = [s.shape[0] for s in seqs]
        hmm = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=rs,
            verbose=self.verbose,
        ).fit(Xc, Ls)
        self._smooth_probs(hmm)
        return hmm

    # ---- required sklearn API ----
    def fit(self, X, y):
        y = np.asarray(y).ravel().astype(int)
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("HMMBinaryClassifier expects binary labels {0,1}.")
        Xc, lengths, seqs = _stack_sequences(X)

        self.scaler_ = StandardScaler() if self.scale else None
        if self.scaler_ is not None:
            Xc = self.scaler_.fit_transform(Xc)
            seqs_scaled, s = [], 0
            for L in lengths:
                seqs_scaled.append(Xc[s:s+L]); s += L
            seqs = seqs_scaled

        pos = [s for s, yi in zip(seqs, y) if yi == 1]
        neg = [s for s, yi in zip(seqs, y) if yi == 0]
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("Both classes must be present.")

        self.hmm_pos_ = self._fit_hmm(pos, rs=0)
        self.hmm_neg_ = self._fit_hmm(neg, rs=1)
        n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
        a = self.prior_smoothing
        self.log_prior_pos_ = np.log((n_pos + a) / (n_pos + n_neg + 2*a))
        self.log_prior_neg_ = np.log((n_neg + a) / (n_pos + n_neg + 2*a))
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def _prep(self, X) -> List[np.ndarray]:
        _, _, seqs = _stack_sequences(X)
        if getattr(self, "scaler_", None) is not None:
            seqs = [self.scaler_.transform(s) for s in seqs]
        return seqs

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, ["hmm_pos_", "hmm_neg_"])
        seqs = self._prep(X)
        lp, ln = [], []
        for s in seqs:
            lp_i = self.hmm_pos_.score(s)
            ln_i = self.hmm_neg_.score(s)
            if self.normalize_ll:
                T = max(len(s), 1)
                lp_i, ln_i = lp_i / T, ln_i / T
            lp.append(lp_i + self.log_prior_pos_)
            ln.append(ln_i + self.log_prior_neg_)
        llr = np.asarray(lp) - np.asarray(ln)
        p1 = expit(llr)  # sigmoid(llr)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
