"""validation_repeated_apparent.py -- what the reported AUC is.

For every staging task and feature family (primary logistic model):
  * reproduces the published out-of-fold (OOF) ROC AUC on the original partition,
    now with 2,000-resample subject-level bootstrap CIs;
  * mean +/- SD of the per-fold AUCs of that partition;
  * stability across 20 independent random 5-fold partitions (mean, SD, 2.5-97.5 pct);
  * apparent (resubstitution) AUC, i.e. the model scored on its own training data,
    to show the optimism that cross-validation removes.
Also stores the OOF probabilities of the primary partition for ROC-curve plotting.
"""
from revision_common import *

X, fam = load_all()
FS = feature_sets(fam)
bio = ["CSF_ABETA42", "CSF_PTAU", "CSF_TAU", "CSF_PTAU_ABETA42", "AMYPET_SUVR", "AMYPET_CENTILOID",
       "TAUPET_METATEMP", "TAUPET_ENTORHINAL", "PLASMA_PTAU217_AB42", "PLASMA_AB42_AB40"]
TASKS = {
    "amyloid": (X["baseline_dx"].eq("MCI") & X["A"].notna(), "A", ["A-", "A+"]),
    "trajectory": (X["traj"].notna(), "traj", ["stable", "slow", "fast"]),
    "fourclass": (X["ordlvl"].notna(), "ordlvl", ["CN", "A-MCI", "A+MCI", "AD"]),
}
N_REP = {"amyloid": 20, "trajectory": 20, "fourclass": 5}
RES = os.path.join(OUT, "A_primary_repeated_apparent.csv")
done = set()
if os.path.exists(RES):
    _d = pd.read_csv(RES); done = set(zip(_d.task, _d.featureset))
for task, (mask, col, classes) in TASKS.items():
    sub = X[mask].reset_index(drop=True)
    y = sub[col].map({1.0: "A+", 0.0: "A-"}).values if col == "A" else sub[col].astype(str).values
    fsets = dict(FS)
    if task == "trajectory":
        fsets["all"] = FS["clinical+fs"] + bio
    base_splits = repeated_kfold_first(y)
    for name, cols in fsets.items():
        if (task, name) in done:
            continue
        cols = [c for c in cols if c in sub.columns]
        P, _ = oof_predict(sub[cols], y, base_splits, "logistic_l2", classes, sub["RID"].values)
        a = auc(y, P, classes); lo, hi = boot_ci(y, P, classes)
        fold_aucs = [auc(y[te], P[te], classes) for _, te in base_splits]
        Pa, _ = apparent_predict(sub[cols], y, "logistic_l2", classes)
        app = auc(y, Pa, classes)
        reps = []
        for r in range(N_REP[task]):
            sp = kfold_splits(y, 5, seed=1000 + r)
            Pr, _ = oof_predict(sub[cols], y, sp, "logistic_l2", classes, sub["RID"].values)
            reps.append(auc(y, Pr, classes))
        reps = np.array(reps)
        row = (dict(task=task, featureset=name, n=len(sub), oof_auc=round(a, 3), ci_lo=round(lo, 3),
                         ci_hi=round(hi, 3), bacc=round(bacc(y, P, classes), 3),
                         perfold_mean=round(np.mean(fold_aucs), 3), perfold_sd=round(np.std(fold_aucs, ddof=1), 3),
                         rep_mean=round(reps.mean(), 3), rep_sd=round(reps.std(ddof=1), 3),
                         rep_p2_5=round(np.percentile(reps, 2.5), 3), rep_p97_5=round(np.percentile(reps, 97.5), 3),
                         n_repeats=len(reps), apparent_auc=round(app, 3), optimism=round(app - a, 3)))
        print(row, flush=True)
        df = pd.DataFrame(P, columns=["p_" + c for c in classes]); df.insert(0, "y_true", y); df.insert(0, "RID", sub["RID"].values)
        df.to_csv(os.path.join(OUT, f"oof_{task}_{name}.csv"), index=False)
        pd.DataFrame([row]).to_csv(RES, mode="a", header=not os.path.exists(RES), index=False)
print("A done")
