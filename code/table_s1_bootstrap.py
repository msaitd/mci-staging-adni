"""table_s1_bootstrap.py -- Supplementary Table S1 recomputed from the stored out-of-fold
predictions of the original run, with 2,000-resample subject-level bootstrap CIs."""
from revision_common import *
from sklearn.metrics import balanced_accuracy_score
W = os.path.join(ROOT, "outputs", "within_mci")
TASK = {"amyloid_mci": ["A-", "A+"], "trajectory": ["stable", "slow", "fast"], "ordinal": ["CN", "A-MCI", "A+MCI", "AD"]}
rows = []
for f in sorted(os.listdir(W)):
    if not f.startswith("oof_"): continue
    stem = f[4:-4]
    task = next(t for t in TASK if stem.startswith(t + "_"))
    rest = stem[len(task) + 1:]
    model = next(m for m in ("logistic_l2", "extra_trees", "hist_gb") if rest.endswith(m))
    fsname = rest[:-(len(model) + 1)]
    d = pd.read_csv(os.path.join(W, f)); cls = TASK[task]
    # same subject order as the revision out-of-fold files, so that bootstrap resamples are identical
    ref = pd.read_csv(os.path.join(OUT, f"oof_{ {'amyloid_mci': 'amyloid', 'ordinal': 'fourclass'}.get(task, task)}_clinical.csv"), usecols=["RID"])
    d = ref.merge(d, on="RID", how="left"); assert d["y_true"].notna().all()
    y = d["y_true"].astype(str).values; P = d[["p_" + c for c in cls]].values
    a = auc(y, P, cls); lo, hi = boot_ci(y, P, cls)
    rows.append(dict(task=task, featureset=fsname, model=model, n=len(d), auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                     bacc=round(balanced_accuracy_score(y, d["y_pred"].astype(str).values), 3)))
t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "G_tableS1_2000boot.csv"), index=False); print(t.to_string(index=False))
