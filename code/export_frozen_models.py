"""export_frozen_models.py -- freeze the ADNI 'portable' models for external validation.

Fits the pre-specified logistic pipeline (median imputation -> standardisation -> L2 logistic
regression, C=1, class-balanced) on the full ADNI cohorts using only
age, sex (female=1), education (years), APOE e4 allele count, MMSE and CDR Sum of Boxes,
and writes the fitted parameters (no subject-level data) to frozen_adni_models.json.

Models: amyloid (A+ vs A- MCI), trajectory (stable/slow/fast MCI) and four-group
(CN / A-MCI / A+MCI / AD dementia). Cross-validated ADNI AUCs of the same pipeline
(primary five-fold partition) are stored for reference.
"""
import json
from revision_common import *
from revision_common import _bin_auc
from ml_common import make_models
from sklearn.base import clone

PORT = ["age", "sex_female", "PTEDUCAT", "APOE4", "MMSE", "CDRSB"]
X, fam = load_all()

TASKS = [
    ("amyloid", X["baseline_dx"].eq("MCI") & X["A"].notna(), lambda s: s["A"].map({1.0: "A+", 0.0: "A-"}), ["A-", "A+"]),
    ("trajectory", X["traj"].notna(), lambda s: s["traj"].astype(str), ["fast", "slow", "stable"]),
    ("fourgroup", X["ordlvl"].notna(), lambda s: s["ordlvl"].astype(str), ["A+MCI", "A-MCI", "AD", "CN"]),
]


def manual_proba(params, Xv):
    Z = np.where(np.isnan(Xv), np.array(params["impute_median"]), Xv)
    Z = (Z - np.array(params["scaler_mean"])) / np.array(params["scaler_scale"])
    coef = np.array(params["coef"]); b = np.array(params["intercept"])
    if len(params["classes"]) == 2:
        p1 = 1 / (1 + np.exp(-(Z @ coef[0] + b[0])))
        return np.column_stack([1 - p1, p1])
    L = Z @ coef.T + b; L -= L.max(axis=1, keepdims=True); E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


out = {"features": PORT,
       "feature_notes": {"age": "years at the clinical visit", "sex_female": "1 = female, 0 = male",
                         "PTEDUCAT": "years of education", "APOE4": "number of APOE e4 alleles (0, 1, 2)",
                         "MMSE": "Mini-Mental State Examination total (0-30)", "CDRSB": "CDR Sum of Boxes (0-18)"},
       "pipeline": "median imputation -> standardisation -> L2 logistic regression (C=1, class_weight=balanced)",
       "source": "ADNI (ADNIMERGE2); fitted on the full analytic cohorts", "models": {}}

for task, mask, ylab, classes in TASKS:
    sub = X[mask].reset_index(drop=True)
    y = ylab(sub).values.astype(str)
    classes = sorted(np.unique(y).tolist())  # sklearn class order (binary: probability of classes[1])
    est = make_models()["logistic_l2"][0]
    # cross-validated reference performance (primary partition)
    P, _ = oof_predict(sub[PORT], y, repeated_kfold_first(y), "logistic_l2", classes, sub["RID"].values)
    a = auc(y, P, classes); lo, hi = boot_ci(y, P, classes)
    # full-cohort fit (the frozen model)
    Xv = sub[PORT].apply(pd.to_numeric, errors="coerce").values
    m = clone(est).fit(Xv, y)
    imp, sc, lr = m.named_steps["imp"], m.named_steps["sc"], m.named_steps["clf"]
    assert list(lr.classes_) == classes, (list(lr.classes_), classes)
    params = {"classes": classes, "n_train": int(len(sub)),
              "class_counts": {c: int((y == c).sum()) for c in classes},
              "impute_median": imp.statistics_.tolist(), "scaler_mean": sc.mean_.tolist(),
              "scaler_scale": sc.scale_.tolist(), "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
              "adni_cv_auc": round(float(a), 3), "adni_cv_auc_ci": [round(float(lo), 3), round(float(hi), 3)]}
    # the manual implementation must reproduce sklearn exactly
    diff = np.abs(manual_proba(params, Xv) - m.predict_proba(Xv)).max()
    assert diff < 1e-9, diff
    out["models"][task] = params
    print(f"{task}: n={len(sub)} classes={params['class_counts']} ADNI CV AUC {a:.3f} ({lo:.3f}-{hi:.3f}); max|diff|={diff:.1e}")

dst = os.path.join(ROOT, "external_validation", "frozen_adni_models.json")
json.dump(out, open(dst, "w"), indent=1)
print("written", os.path.abspath(dst))
