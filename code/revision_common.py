"""revision_common.py -- shared loading and leakage-safe helpers for the revision analyses.

All models are the same fixed-hyper-parameter pipelines as ml_common.make_models():
imputation and scaling are fitted inside each training fold only. Every modelling
table has one row per subject, so any split is a subject-level split; disjointness of
train/test subjects is asserted for every fold.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from ml_common import make_models  # noqa: E402

D = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs", "revision"); os.makedirs(OUT, exist_ok=True)
SEED = 42
N_BOOT = 2000


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------
def load_all():
    m = pd.read_csv(os.path.join(D, "master_features.csv"), low_memory=False)
    b = pd.read_csv(os.path.join(D, "biomarkers_baseline.csv"))
    det = pd.read_csv(os.path.join(D, "amyloid_detail_baseline.csv"))
    conv = pd.read_csv(os.path.join(D, "cohort_conversion.csv"), low_memory=False)
    fam = json.load(open(os.path.join(D, "feature_families.json")))
    X = m.merge(b, on="RID", how="left").merge(det, on="RID", how="left")
    X["baseline_date"] = pd.to_datetime(X["baseline_date"], errors="coerce")
    for c in ("PET_DATE", "CSF_DATE"):
        X[c] = pd.to_datetime(X[c], errors="coerce")
    X["site"] = X["PTID"].astype(str).str[:3]
    # primary amyloid label (unchanged from the original analysis)
    X["A"] = primary_label(X)
    X["A_source"] = np.where(X["AMYPET_CENTILOID"].notna(), "PET",
                             np.where(X["CSF_ABETA42"].notna(), "CSF", None))
    # months between the label-defining measurement and the clinical baseline
    lab_date = np.where(X["A_source"] == "PET", X["PET_DATE"],
                        np.where(X["A_source"] == "CSF", X["CSF_DATE"], pd.NaT))
    X["A_dt_months"] = (pd.to_datetime(lab_date) - X["baseline_date"]).dt.days / 30.44
    # trajectory label (unchanged)
    cv = conv[["RID", "first_AD_month", "max_followup_m"]].copy()
    def traj(r):
        fa, fu = r["first_AD_month"], r["max_followup_m"]
        if pd.notna(fa):
            return "fast" if fa <= 24 else ("slow" if fa <= 48 else np.nan)
        return "stable" if (pd.notna(fu) and fu >= 48) else np.nan
    cv["traj"] = cv.apply(traj, axis=1)
    X = X.merge(cv[["RID", "traj", "first_AD_month"]], on="RID", how="left")
    def ordlvl(r):
        dx = r["baseline_dx"]
        if dx == "CN": return "CN"
        if dx == "AD": return "AD"
        if dx == "MCI":
            return "A-MCI" if r["A"] == 0 else ("A+MCI" if r["A"] == 1 else np.nan)
        return np.nan
    X["ordlvl"] = X.apply(ordlvl, axis=1)
    return X, fam


def primary_label(X, cl_cut=20.0, csf_cut=980.0):
    pet = X["AMYPET_CENTILOID"]; csf = X["CSF_ABETA42"]
    out = np.where(pet.notna(), (pet >= cl_cut).astype(float),
                   np.where(csf.notna(), (csf < csf_cut).astype(float), np.nan))
    return pd.Series(out, index=X.index)


def feature_sets(fam):
    demo, cog, fs = fam["demo"], fam["cognition"], fam["freesurfer"]
    return {"demo_apoe": demo, "cognition": cog, "freesurfer": fs,
            "clinical": demo + cog, "clinical+fs": demo + cog + fs}


# ----------------------------------------------------------------------------------
# cross-validated prediction with arbitrary (subject-level) splits
# ----------------------------------------------------------------------------------
def kfold_splits(y, n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def repeated_kfold_first(y, n_splits=5, seed=SEED):
    """Exactly the partition used in the original analysis (first repeat of
    RepeatedStratifiedKFold with random_state=42)."""
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=seed)
    return list(rkf.split(np.zeros(len(y)), y))


def group_splits(groups):
    """Leave-one-group-out splits (e.g. site or ADNI phase)."""
    g = np.asarray(groups); out = []
    for val in pd.unique(g):
        te = np.where(g == val)[0]; tr = np.where(g != val)[0]
        out.append((tr, te))
    return out


def oof_predict(Xdf, y, splits, model="logistic_l2", classes=None, rid=None):
    """Out-of-fold class probabilities. Preprocessing inside the fold pipeline."""
    est = make_models()[model][0]
    Xv = Xdf.apply(pd.to_numeric, errors="coerce").values
    y = np.asarray(y).astype(str)
    classes = classes or sorted(np.unique(y))
    P = np.full((len(y), len(classes)), np.nan)
    rid = np.arange(len(y)) if rid is None else np.asarray(rid)
    for tr, te in splits:
        assert not (set(rid[tr]) & set(rid[te])), "subject overlap train/test"
        if len(np.unique(y[tr])) < len(classes):
            continue  # a training fold lacking a class cannot be scored
        mdl = clone(est).fit(Xv[tr], y[tr])
        cl = list(mdl.classes_)
        P[te] = mdl.predict_proba(Xv[te])[:, [cl.index(c) for c in classes]]
    return P, classes


def apparent_predict(Xdf, y, model="logistic_l2", classes=None):
    """Resubstitution (training-set) probabilities -- optimistic by construction."""
    est = make_models()[model][0]
    Xv = Xdf.apply(pd.to_numeric, errors="coerce").values
    y = np.asarray(y).astype(str); classes = classes or sorted(np.unique(y))
    mdl = clone(est).fit(Xv, y); cl = list(mdl.classes_)
    return mdl.predict_proba(Xv)[:, [cl.index(c) for c in classes]], classes


def _bin_auc(pos, score):
    """Mann-Whitney AUC with average ranks for ties (identical to roc_auc_score)."""
    from scipy.stats import rankdata
    pos = np.asarray(pos, bool); n1 = pos.sum(); n0 = len(pos) - n1
    r = rankdata(score)
    return (r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auc(y, P, classes):
    """Binary ROC AUC, or macro-averaged one-vs-rest ROC AUC for >2 classes."""
    y = np.asarray(y).astype(str)
    ok = ~np.isnan(P).any(axis=1)
    y, P = y[ok], P[ok]
    if len(classes) == 2:
        return _bin_auc(y == classes[1], P[:, 1])
    return float(np.mean([_bin_auc(y == c, P[:, k]) for k, c in enumerate(classes)]))


def bacc(y, P, classes):
    y = np.asarray(y).astype(str); ok = ~np.isnan(P).any(axis=1)
    pred = np.array(classes)[P[ok].argmax(1)]
    return balanced_accuracy_score(y[ok], pred)


def boot_ci(y, P, classes, n=N_BOOT, seed=1):
    y = np.asarray(y).astype(str); ok = ~np.isnan(P).any(axis=1)
    y, P = y[ok], P[ok]
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < len(classes): continue
        vals.append(auc(y[s], P[s], classes))
    return np.percentile(vals, [2.5, 97.5])


def paired_delta(y, Pa, Pb, classes, n=N_BOOT, seed=7):
    """Paired subject-level bootstrap of AUC(a) - AUC(b)."""
    y = np.asarray(y).astype(str)
    ok = ~(np.isnan(Pa).any(axis=1) | np.isnan(Pb).any(axis=1))
    y, Pa, Pb = y[ok], Pa[ok], Pb[ok]
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < len(classes): continue
        d.append(auc(y[s], Pa[s], classes) - auc(y[s], Pb[s], classes))
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return auc(y, Pa, classes) - auc(y, Pb, classes), lo, hi, min(1.0, p)


def fmt_ci(v, lo, hi, k=3):
    return f"{v:.{k}f} ({lo:.{k}f}–{hi:.{k}f})"
