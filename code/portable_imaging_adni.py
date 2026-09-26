"""portable_imaging_adni.py -- ADNI counterpart of the OASIS-3 imaging comparison.

Adds the compact AD-signature summary (5) or the full FreeSurfer panel (324) to the six-variable
portable model (age, sex, education, APOE e4, MMSE, CDR-SB) in ADNI, with the pre-specified logistic
pipeline and the primary five-fold partition; paired subject-level bootstrap of the AUC difference.
Output: outputs/revision/J_portable_imaging.csv
"""
from revision_common import *
import atrophy_defs as E

PORT = ["age", "sex_female", "PTEDUCAT", "APOE4", "MMSE", "CDRSB"]
X, fam = load_all()
X = E.add_compact(X)
FS = fam["freesurfer"]
rows = []
for task, mask, col in (("amyloid", X["baseline_dx"].eq("MCI") & X["A"].notna(), "A"),
                        ("trajectory", X["traj"].notna(), "traj")):
    sub = X[mask].reset_index(drop=True)
    y = (sub["A"].map({1.0: "A+", 0.0: "A-"}) if col == "A" else sub["traj"]).astype(str).values
    classes = sorted(np.unique(y).tolist()); spl = repeated_kfold_first(y)
    P = {}
    for name, cols in (("portable", PORT), ("portable + compact (5)", PORT + E.COMPACT),
                       ("portable + full FreeSurfer (324)", PORT + FS)):
        P[name], _ = oof_predict(sub[cols], y, spl, "logistic_l2", classes, sub["RID"].values)
        a = auc(y, P[name], classes); lo, hi = boot_ci(y, P[name], classes)
        rows.append(dict(task=task, model=name, n=len(sub), auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
    for name in ("portable + compact (5)", "portable + full FreeSurfer (324)"):
        d, lo, hi, p = paired_delta(y, P[name], P["portable"], classes)
        rows.append(dict(task=task, model=f"delta: {name} minus portable", n=len(sub), auc=round(d, 3),
                         ci_lo=round(lo, 3), ci_hi=round(hi, 3), p=round(p, 4)))
res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, "J_portable_imaging.csv"), index=False)
print(res.to_string(index=False))
