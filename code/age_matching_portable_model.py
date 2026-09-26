"""age_matching_portable_model.py -- age matching and a portable model for transportability.

(1) Age-matched amyloid analysis: amyloid-negative MCI are matched 1:1 to amyloid-positive
    MCI on age (greedy nearest neighbour without replacement, caliper 1 year, random order,
    seed 42). Characteristics and staging performance are re-estimated in the matched set,
    with and without age as a predictor.
(2) Age dependence in the full cohort: clinical model without age; AUC within age tertiles.
(3) "Portable" model restricted to predictors that are also collected in other cohorts
    (age, sex, education, APOE e4, MMSE, CDR-SB), to facilitate external validation.
"""
from revision_common import *
from scipy import stats
import atrophy_defs as E

X, fam = load_all(); X = E.add_compact(X)
demo, cog, fs = fam["demo"], fam["cognition"], fam["freesurfer"]
M = X[X["baseline_dx"].eq("MCI") & X["A"].notna()].reset_index(drop=True)

# ---------------- (1) age matching ----------------
rng = np.random.default_rng(42)
neg = M[M["A"] == 0].sample(frac=1, random_state=42); pos = M[M["A"] == 1].copy()
avail = set(pos.index); pairs = []
for i, r in neg.iterrows():
    if not avail: break
    cand = pos.loc[list(avail)]; d = (cand["age"] - r["age"]).abs()
    j = d.idxmin()
    if d[j] <= 1.0:
        pairs.append((i, j)); avail.remove(j)
idx = [i for p in pairs for i in p]
MM = M.loc[idx].reset_index(drop=True)
chars = []
for col, lab, kind in [("age", "Age, years", "c"), ("sex_female", "Female", "b"), ("PTEDUCAT", "Education, years", "c"),
                       ("APOE4", "APOE e4 carrier", "b"), ("MMSE", "MMSE", "c"), ("CDRSB", "CDR-SB", "c"), ("ADAS13", "ADAS-Cog13", "c")]:
    a = pd.to_numeric(MM.loc[MM.A == 1, col], errors="coerce"); b = pd.to_numeric(MM.loc[MM.A == 0, col], errors="coerce")
    if kind == "b":
        ta, tb = (a >= 1).sum(), (b >= 1).sum()
        p = stats.chi2_contingency([[ta, a.notna().sum() - ta], [tb, b.notna().sum() - tb]])[1]
        chars.append((lab, f"{100 * ta / a.notna().sum():.0f}%", f"{100 * tb / b.notna().sum():.0f}%", f"{p:.3g}"))
    else:
        p = stats.ttest_ind(a, b, nan_policy="omit", equal_var=False).pvalue
        chars.append((lab, f"{a.mean():.1f} ± {a.std():.1f}", f"{b.mean():.1f} ± {b.std():.1f}", f"{p:.3g}"))
ch = pd.DataFrame(chars, columns=["characteristic", "A+ MCI (matched)", "A- MCI (matched)", "p"])
ch.to_csv(os.path.join(OUT, "F_age_matched_characteristics.csv"), index=False)
print("matched pairs:", len(pairs)); print(ch.to_string(index=False))

no_age = [c for c in demo if c != "age"]
SETS = {"APOE e4 only": ["APOE4"], "demographics+APOE": demo, "clinical": demo + cog,
        "clinical without age": no_age + cog, "freesurfer": fs, "clinical+freesurfer": demo + cog + fs,
        "clinical+compact atrophy": demo + cog + E.COMPACT}
rows = []
y = np.where(MM["A"] == 1, "A+", "A-"); cls = ["A-", "A+"]; sp = repeated_kfold_first(y); P = {}
for name, cols in SETS.items():
    P[name], _ = oof_predict(MM[cols], y, sp, "logistic_l2", cls, MM["RID"].values)
    a = auc(y, P[name], cls); lo, hi = boot_ci(y, P[name], cls)
    rows.append(dict(cohort=f"age-matched (n={len(MM)})", featureset=name, auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
d, lo, hi, p = paired_delta(y, P["clinical"], P["clinical+freesurfer"], cls)
rows.append(dict(cohort=f"age-matched (n={len(MM)})", featureset="delta clinical - (clinical+freesurfer)",
                 auc=round(d, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3), p=round(p, 4)))

# ---------------- (2) full cohort: without age, age tertiles ----------------
yF = np.where(M["A"] == 1, "A+", "A-"); spF = repeated_kfold_first(yF)
for name in ("clinical", "clinical without age"):
    PF, _ = oof_predict(M[SETS[name]], yF, spF, "logistic_l2", cls, M["RID"].values)
    a = auc(yF, PF, cls); lo, hi = boot_ci(yF, PF, cls)
    rows.append(dict(cohort="full (n=1059)", featureset=name, auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
    if name == "clinical":
        tert = pd.qcut(M["age"], 3, labels=["lowest tertile", "middle tertile", "highest tertile"])
        for t in tert.cat.categories:
            k = (tert == t).values
            a = auc(yF[k], PF[k], cls); lo, hi = boot_ci(yF[k], PF[k], cls)
            rng_ = f"{M.loc[k, 'age'].min():.0f}-{M.loc[k, 'age'].max():.0f} y"
            rows.append(dict(cohort=f"full, age {t} ({rng_}, n={int(k.sum())}, A+ {100 * (yF[k] == 'A+').mean():.0f}%)",
                             featureset="clinical", auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))

# ---------------- (3) portable model ----------------
PORT = ["age", "sex_female", "PTEDUCAT", "APOE4", "MMSE", "CDRSB"]
for task, mask, col, classes in [("amyloid", X["baseline_dx"].eq("MCI") & X["A"].notna(), "A", ["A-", "A+"]),
                                 ("trajectory", X["traj"].notna(), "traj", ["stable", "slow", "fast"])]:
    sub = X[mask].reset_index(drop=True)
    yy = sub[col].map({1.0: "A+", 0.0: "A-"}).values if col == "A" else sub[col].astype(str).values
    for dname, spl in (("5-fold (primary partition)", repeated_kfold_first(yy)), ("LOSO", group_splits(sub["site"].values))):
        PP, _ = oof_predict(sub[PORT], yy, spl, "logistic_l2", classes, sub["RID"].values)
        a = auc(yy, PP, classes); lo, hi = boot_ci(yy, PP, classes)
        rows.append(dict(cohort=f"{task}, {dname} (n={len(sub)})", featureset="portable (age, sex, education, APOE e4, MMSE, CDR-SB)",
                         auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, "F_age_portable.csv"), index=False)
print(res.to_string(index=False)); print("F done")
