"""atrophy_compact_robustness.py -- robustness of the compact AD-signature atrophy increment.

For amyloid, trajectory and four-class staging: clinical vs clinical + compact atrophy
(5 measures) under (i) 20 random 5-fold partitions, (ii) leave-one-site-out and temporal
(ADNI-1/GO/2 -> ADNI-3/4) validation, and (iii) the two secondary classifiers.
"""
from revision_common import *
import atrophy_defs as E  # compact-measure definitions shared with atrophy_redundancy

X, fam = load_all(); X = E.add_compact(X)
demo, cog = fam["demo"], fam["cognition"]; COMPACT = E.COMPACT
TASKS = {"amyloid": (X["baseline_dx"].eq("MCI") & X["A"].notna(), "A", ["A-", "A+"]),
         "trajectory": (X["traj"].notna(), "traj", ["stable", "slow", "fast"]),
         "fourclass": (X["ordlvl"].notna(), "ordlvl", ["CN", "A-MCI", "A+MCI", "AD"])}
REPS = {"amyloid": 20, "trajectory": 20, "fourclass": 5}
rows = []
for task, (mask, col, classes) in TASKS.items():
    sub = X[mask].reset_index(drop=True)
    y = sub[col].map({1.0: "A+", 0.0: "A-"}).values if col == "A" else sub[col].astype(str).values
    rid = sub["RID"].values
    def run(sp, model="logistic_l2"):
        Pc, _ = oof_predict(sub[demo + cog], y, sp, model, classes, rid)
        Pa, _ = oof_predict(sub[demo + cog + COMPACT], y, sp, model, classes, rid)
        return Pc, Pa
    # (i) repeated partitions
    ds = []
    for r in range(REPS[task]):
        Pc, Pa = run(kfold_splits(y, 5, seed=2000 + r)); ds.append(auc(y, Pa, classes) - auc(y, Pc, classes))
    ds = np.array(ds)
    rows.append(dict(task=task, design=f"{REPS[task]} random 5-fold partitions", model="logistic_l2",
                     auc_clinical="", auc_clinical_compact="", delta=round(ds.mean(), 3),
                     ci_lo=round(np.percentile(ds, 2.5), 3), ci_hi=round(np.percentile(ds, 97.5), 3),
                     p=f"{(ds > 0).mean():.2f} of partitions > 0"))
    print(rows[-1], flush=True)
    # (ii) non-random splits and (iii) other classifiers
    early = sub["phase"].isin(["ADNI1", "ADNIGO", "ADNI2"]).values
    designs = [("LOSO", group_splits(sub["site"].values), "logistic_l2"),
               ("temporal", [(np.where(early)[0], np.where(~early)[0])], "logistic_l2"),
               ("5-fold (primary partition)", repeated_kfold_first(y), "extra_trees"),
               ("5-fold (primary partition)", repeated_kfold_first(y), "hist_gb")]
    if task == "fourclass":
        designs = designs[1:2] + designs[2:]
    for dname, sp, model in designs:
        Pc, Pa = run(sp, model)
        d, lo, hi, p = paired_delta(y, Pa, Pc, classes)
        rows.append(dict(task=task, design=dname, model=model, auc_clinical=round(auc(y, Pc, classes), 3),
                         auc_clinical_compact=round(auc(y, Pa, classes), 3), delta=round(d, 3),
                         ci_lo=round(lo, 3), ci_hi=round(hi, 3), p=round(p, 4)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "E2_compact_robustness.csv"), index=False)
print("E2 done")
