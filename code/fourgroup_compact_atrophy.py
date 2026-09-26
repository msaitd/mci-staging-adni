"""fourgroup_compact_atrophy.py -- four-class staging with the compact AD-signature panel
(primary partition, pre-specified logistic model); appended to E_atrophy_models.csv."""
from revision_common import *
import atrophy_defs as E
X, fam = load_all(); X = E.add_compact(X)
demo, cog = fam["demo"], fam["cognition"]
sub = X[X["ordlvl"].notna()].reset_index(drop=True); y = sub["ordlvl"].astype(str).values
cls = ["CN", "A-MCI", "A+MCI", "AD"]; sp = repeated_kfold_first(y)
P, _ = oof_predict(sub[demo + cog + E.COMPACT], y, sp, "logistic_l2", cls, sub["RID"].values)
Pc, _ = oof_predict(sub[demo + cog], y, sp, "logistic_l2", cls, sub["RID"].values)
a = auc(y, P, cls); lo, hi = boot_ci(y, P, cls); d, dlo, dhi, p = paired_delta(y, P, Pc, cls)
f = os.path.join(OUT, "E_atrophy_models.csv"); t = pd.read_csv(f)
t = t[~((t.task == "fourclass") & (t.featureset == "clinical+compact_raw"))]
t = pd.concat([t, pd.DataFrame([dict(task="fourclass", featureset="clinical+compact_raw", n=len(sub), n_features=15,
                                     auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3))])])
t.to_csv(f, index=False)
g = os.path.join(OUT, "E_atrophy_delta.csv"); u = pd.read_csv(g)
u = u[~((u.task == "fourclass"))]
u = pd.concat([u, pd.DataFrame([dict(task="fourclass", comparison="clinical+compact_raw - clinical", delta=round(d, 3),
                                     ci_lo=round(dlo, 3), ci_hi=round(dhi, 3), p=round(p, 4))])])
u.to_csv(g, index=False); print(a, lo, hi, d, dlo, dhi, p)
