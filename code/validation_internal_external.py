"""validation_internal_external.py -- heterogeneity / transportability.

Internal-external validation within ADNI (non-random splits):
  * leave-one-site-out (LOSO): each recruitment/acquisition site is held out in turn;
  * leave-one-ADNI-phase-out: each protocol phase (ADNI-1, GO, 2, 3, 4) held out in turn;
  * temporal split: train on ADNI-1/GO/2, test on the later ADNI-3/4 enrolment.
The same fixed, pre-specified pipelines are used; preprocessing is fitted on the
training partition only. Held-out probabilities are cached (resumable); summaries:
pooled held-out AUC with bootstrap CI, per-site and per-phase AUCs, paired AUC
difference clinical minus clinical+FreeSurfer, and temporal-split calibration.
"""
import time
from revision_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

T0 = time.time(); BUDGET = float(os.environ.get("BUDGET", "150"))
CACHE = os.path.join(OUT, "cache_B"); os.makedirs(CACHE, exist_ok=True)
X, fam = load_all()
FS = feature_sets(fam)
TASKS = {
    "amyloid": (X["baseline_dx"].eq("MCI") & X["A"].notna(), "A", ["A-", "A+"]),
    "trajectory": (X["traj"].notna(), "traj", ["stable", "slow", "fast"]),
    "fourclass": (X["ordlvl"].notna(), "ordlvl", ["CN", "A-MCI", "A+MCI", "AD"]),
}
SETS = {"amyloid": ["demo_apoe", "freesurfer", "clinical", "clinical+fs"],
        "trajectory": ["freesurfer", "clinical", "clinical+fs"],
        "fourclass": ["clinical", "clinical+fs"]}
DESIGNS = ["LOSO", "leave-phase-out", "temporal"]


def data(task):
    mask, col, classes = TASKS[task]
    sub = X[mask].reset_index(drop=True)
    y = sub[col].map({1.0: "A+", 0.0: "A-"}).values if col == "A" else sub[col].astype(str).values
    return sub, y, classes


def splits(sub, design):
    if design == "LOSO":
        return group_splits(sub["site"].values)
    if design == "leave-phase-out":
        return group_splits(sub["phase"].values)
    early = sub["phase"].isin(["ADNI1", "ADNIGO", "ADNI2"]).values
    return [(np.where(early)[0], np.where(~early)[0])]


# ---------------- stage 1: cached held-out predictions ----------------
todo = [(t, d, f) for t in TASKS for d in DESIGNS for f in SETS[t]]
for task, design, fsname in todo:
    fn = os.path.join(CACHE, f"{task}__{design}__{fsname}.npy")
    if os.path.exists(fn):
        continue
    if time.time() - T0 > BUDGET:
        print("BUDGET reached -- rerun to continue"); sys.exit(0)
    sub, y, classes = data(task)
    cols = [c for c in FS[fsname] if c in sub.columns]
    P, _ = oof_predict(sub[cols], y, splits(sub, design), "logistic_l2", classes, sub["RID"].values)
    np.save(fn, P); print("cached", task, design, fsname, round(time.time() - T0), "s", flush=True)

# ---------------- stage 2: summaries ----------------
rows, site_rows, phase_rows, delta_rows, cal_rows = [], [], [], [], []
for task in TASKS:
    sub, y, classes = data(task)
    for design in DESIGNS:
        preds = {f: np.load(os.path.join(CACHE, f"{task}__{design}__{f}.npy")) for f in SETS[task]}
        for fsname, P in preds.items():
            ok = ~np.isnan(P).any(axis=1)
            a = auc(y[ok], P[ok], classes); lo, hi = boot_ci(y[ok], P[ok], classes)
            rows.append(dict(task=task, design=design, featureset=fsname, n_scored=int(ok.sum()),
                             auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
            if design == "LOSO" and task == "amyloid":
                for s in pd.unique(sub["site"]):
                    k = (sub["site"].values == s) & ok
                    if k.sum() >= 10 and len(np.unique(y[k])) == 2:
                        site_rows.append(dict(task=task, featureset=fsname, site=s, n=int(k.sum()),
                                              auc=auc(y[k], P[k], classes)))
            if design == "leave-phase-out":
                for ph in ["ADNI1", "ADNIGO", "ADNI2", "ADNI3", "ADNI4"]:
                    k = (sub["phase"].values == ph) & ok
                    if k.sum() and len(np.unique(y[k])) == len(classes):
                        phase_rows.append(dict(task=task, featureset=fsname, phase=ph, n=int(k.sum()),
                                               auc=round(auc(y[k], P[k], classes), 3)))
            if design == "temporal" and task == "amyloid" and fsname in ("clinical", "clinical+fs"):
                yt = (y[ok] == "A+").astype(int); p = np.clip(P[ok][:, 1], 1e-6, 1 - 1e-6)
                lg = np.log(p / (1 - p)).reshape(-1, 1)
                lr = LogisticRegression(C=1e6).fit(lg, yt)
                cal_rows.append(dict(task=task, design=design, featureset=fsname, n=int(ok.sum()),
                                     brier=round(brier_score_loss(yt, p), 3),
                                     cal_slope=round(float(lr.coef_[0, 0]), 2),
                                     cal_intercept=round(float(lr.intercept_[0]), 2),
                                     observed_prev=round(yt.mean(), 3), mean_pred=round(p.mean(), 3)))
        if "clinical+fs" in preds:
            d, lo, hi, p = paired_delta(y, preds["clinical"], preds["clinical+fs"], classes)
            delta_rows.append(dict(task=task, design=design, comparison="clinical - (clinical+FS)",
                                   delta=round(d, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3), p=round(p, 4)))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "B_internal_external.csv"), index=False)
sr = pd.DataFrame(site_rows); sr.to_csv(os.path.join(OUT, "B_loso_per_site.csv"), index=False)
sr.groupby("featureset").auc.describe(percentiles=[.25, .5, .75]).round(3).to_csv(
    os.path.join(OUT, "B_loso_per_site_summary.csv"))
pd.DataFrame(phase_rows).to_csv(os.path.join(OUT, "B_per_phase.csv"), index=False)
pd.DataFrame(delta_rows).to_csv(os.path.join(OUT, "B_delta.csv"), index=False)
pd.DataFrame(cal_rows).to_csv(os.path.join(OUT, "B_temporal_calibration.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(delta_rows).to_string(index=False))
print(pd.DataFrame(cal_rows).to_string(index=False))
print("B done")
