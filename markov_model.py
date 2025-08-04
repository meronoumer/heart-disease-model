import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    f1_score,
    average_precision_score
)
from joblib import dump

# === Data Loading ===
df = pd.read_csv("extracted_features_df.csv")

# === Parse `mfcc devation` strings into numeric columns (Option B) ===
def parse_mfcc_dev(s: str) -> list[float]:
    content = s.strip().lstrip('[').rstrip(']')
    return [float(x) for x in content.split()]

parsed    = df['mfcc devation'].apply(parse_mfcc_dev)
n_feats   = len(parsed.iloc[0])
mfcc_cols = [f"mfcc_dev_{i}" for i in range(n_feats)]
df_mfcc   = pd.DataFrame(parsed.tolist(), columns=mfcc_cols, index=df.index)
df        = pd.concat([df.drop(columns=['mfcc devation']), df_mfcc], axis=1)

# === Constants ===
SEQ_KEY     = "file_key"
PATIENT_KEY = "patient_id_x"
META_COLS   = ['Unnamed: 0','patient_id_x','file_key','audio_filename_base','Age','Gender','Smoker','Lives']
LABEL_COLS  = ["AS","AR","MR","MS","N"]
LABELS      = LABEL_COLS
assert set(LABEL_COLS).issubset(df.columns), "Label cols missing"

# === Preprocessing ===
FEATURE_COLS = [c for c in df.columns if c not in META_COLS + LABEL_COLS]
X_raw        = df[FEATURE_COLS].astype(float)
imp          = SimpleImputer(strategy="median")
X_imp        = imp.fit_transform(X_raw)
scaler       = StandardScaler()
X_scaled     = scaler.fit_transform(X_imp)

# === Sequence & Label Assembly ===
groups       = df.groupby(SEQ_KEY, sort=False)
X_seqs, patient_ids, lengths, labels_list = [], [], [], []
for fk, grp in groups:
    idxs = grp.index.to_numpy()
    X_seqs.append(X_scaled[idxs])
    patient_ids.append(grp[PATIENT_KEY].iloc[0])
    lengths.append(len(idxs))
    labels_list.append(grp[LABEL_COLS].iloc[0].values)
Y_seq       = np.vstack(labels_list)
strat_label = Y_seq.argmax(axis=1)
patient_ids = np.array(patient_ids)

# === CV Setup ===
outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# === HMM Helpers ===
def smooth_hmm_probs(model, epsilon=1e-6):
    tm = np.nan_to_num(model.transmat_, nan=0.0) + epsilon
    tm = tm / tm.sum(axis=1, keepdims=True)
    model.transmat_  = tm
    sp = np.nan_to_num(model.startprob_, nan=0.0) + epsilon
    model.startprob_ = sp / sp.sum()

def concat_seqs(idxs):
    X    = np.vstack([X_seqs[i] for i in idxs])
    lens = [len(X_seqs[i])      for i in idxs]
    return X, lens

def fit_hmm(n_states, X, lengths, rs=0):
    m = GaussianHMM(n_components=n_states,
                    covariance_type="diag",
                    n_iter=200,
                    tol=1e-3,
                    random_state=rs)
    m.fit(X, lengths)
    smooth_hmm_probs(m, epsilon=1e-6)
    return m

def smooth_transitions(model, epsilon=1e-6):
    mat = model.transmat_ + epsilon
    model.transmat_  = mat / mat.sum(axis=1, keepdims=True)
    sp  = model.startprob_ + epsilon
    model.startprob_ = sp / sp.sum()

def train_pos_neg_hmms(label, idxs, k, downsample_neg=True):
    lab_col = LABELS.index(label)
    pos_ids = [i for i in idxs if Y_seq[i,lab_col]==1]
    neg_ids = [i for i in idxs if Y_seq[i,lab_col]==0]
    if downsample_neg and len(neg_ids) > len(pos_ids):
        np.random.seed(0)
        neg_ids = list(np.random.choice(neg_ids, len(pos_ids), replace=False))

    Xp, lp   = concat_seqs(pos_ids)
    pos_hmm  = fit_hmm(k, Xp, lp, rs=0)
    smooth_transitions(pos_hmm)

    Xn, ln   = concat_seqs(neg_ids)
    neg_hmm  = fit_hmm(k, Xn, ln, rs=1)
    smooth_transitions(neg_hmm)

    return pos_hmm, neg_hmm

def delta_logl_score(pos_hmm, neg_hmm, X_seq):
    T = len(X_seq)
    return (pos_hmm.score(X_seq) - neg_hmm.score(X_seq)) / T

# === Patched evaluate_k ===
def evaluate_k(idxs, label, k):
    lab_col = LABELS.index(label)
    f1s, aps = [], []

    for tr_i, val_i in inner_cv.split(idxs, [Y_seq[i,lab_col] for i in idxs]):
        tr_ids  = [idxs[i] for i in tr_i]
        val_ids = [idxs[i] for i in val_i]
        pos_hmm, neg_hmm = train_pos_neg_hmms(label, tr_ids, k)

        # gather true labels & scores in arrays
        y_true = np.array([int(Y_seq[i,lab_col]) for i in val_ids])
        y_scr  = np.array([delta_logl_score(pos_hmm, neg_hmm, X_seqs[i])
                           for i in val_ids])

        # drop any NaN or infinite scores
        mask   = np.isfinite(y_scr)
        y_true = y_true[mask]
        y_scr  = y_scr[mask]

        # skip folds with <2 samples or only one class
        if y_true.size < 2 or np.unique(y_true).size < 2:
            continue

        f1s.append(f1_score(y_true, y_scr >= 0))
        aps.append(average_precision_score(y_true, y_scr))

    # if every inner fold was skipped, return zero
    if not f1s:
        return 0.0, 0.0

    return np.mean(f1s), np.mean(aps)

def select_states_label_nested(train_idx, label, cand_states=range(3,11)):
    results = {k: evaluate_k(train_idx, label, k) for k in cand_states}
    best_k  = max(results, key=lambda kk: results[kk][1])
    return best_k, results

# === Entrypoint ===
def main():
    all_scores = {lab: [] for lab in LABELS}
    all_truths = {lab: [] for lab in LABELS}

    for fold, (tr, te) in enumerate(
            outer_cv.split(np.arange(len(X_seqs)), strat_label, groups=patient_ids), 1):
        print(f"\nStarting fold {fold}…", flush=True)

        # choose k for each label
        k_per_label = {
            lab: select_states_label_nested(tr, lab)[0]
            for lab in LABELS
        }

        # train one pos/neg HMM per label
        pos_neg_models = {
            lab: train_pos_neg_hmms(lab, tr, k_per_label[lab])
            for lab in LABELS
        }

        # score test set
        for i in te:
            for lab in LABELS:
                all_truths[lab].append(int(Y_seq[i, LABELS.index(lab)]))
                pos_hmm, neg_hmm = pos_neg_models[lab]
                all_scores[lab].append(delta_logl_score(pos_hmm, neg_hmm, X_seqs[i]))

        print(f"Fold {fold} done; k_per_label={k_per_label}", flush=True)

    # Thresholding & Evaluation
    thresholds = {}
    for lab in LABELS:
        y_t = np.array(all_truths[lab])
        y_s = np.array(all_scores[lab])
        mask = np.isfinite(y_s)
        y_t, y_s = y_t[mask], y_s[mask]

        prec, rec, thr = precision_recall_curve(y_t, y_s)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        thresholds[lab] = thr[f1s.argmax()]

    print("Thresholds:", thresholds)

    y_true_all = np.vstack([all_truths[lab] for lab in LABELS]).T
    y_scr_all  = np.column_stack([all_scores[lab] for lab in LABELS])
    y_pred_all = (y_scr_all >= np.array([thresholds[lab] for lab in LABELS])).astype(int)

    print("Micro F1:", f1_score(y_true_all, y_pred_all, average="micro"))
    print("Macro F1:", f1_score(y_true_all, y_pred_all, average="macro"))
    print(classification_report(y_true_all, y_pred_all, target_names=LABELS))

    # Final Refit & Save
    k_final      = {lab: select_states_label_nested(np.arange(len(X_seqs)), lab)[0]
                    for lab in LABELS}
    final_models = {
        lab: train_pos_neg_hmms(lab, np.arange(len(X_seqs)), k_final[lab])
        for lab in LABELS
    }

    dump({
        "imputer":      imp,
        "scaler":       scaler,
        "models":       final_models,
        "thresholds":   thresholds,
        "k_per_label":  k_final,
        "feature_cols": FEATURE_COLS,
        "labels":       LABELS
    }, "hmm_multilabel_pipeline.joblib")

if __name__ == "__main__":
    main()
