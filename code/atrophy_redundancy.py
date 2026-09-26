"""atrophy_redundancy.py -- why does structural atrophy not add incremental value?

(1) Age/sex-adjusted atrophy (W-scores). Every FreeSurfer measure is expressed as a
    W-score relative to amyloid-negative cognitively normal participants
    (W = (observed - expected[age, sex]) / residual SD). The reference group is disjoint
    from the MCI cohort, so no information from MCI test folds is used.
(2) Compact AD-signature panel (5 measures: hippocampal and amygdala volume, entorhinal
    thickness, AD-signature cortical thickness [entorhinal, inferior/middle temporal,
    fusiform], lateral + inferior-lateral ventricular volume), raw and W-scored, to remove
    the high-dimensionality explanation.
(3) Redundancy decomposition: (a) variance of each atrophy measure explained by age, sex,
    APOE and cognition; (b) correlation of FreeSurfer-only and clinical out-of-fold risk;
    (c) unique atrophy information: atrophy residualised on the clinical predictors inside
    each training fold, then used alone to predict the outcome; (d) in-sample likelihood-
    ratio test of the conditional association (clinical vs clinical + compact atrophy).
(4) Ageing: early (0-12 month) atrophy rates by amyloid status adjusted for age, sex and
    the baseline value of the measure (ANCOVA).
"""
from revision_common import *
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression

X, fam = load_all()
FS = feature_sets(fam); fs_cols = fam["freesurfer"]; demo, cog = fam["demo"], fam["cognition"]
num = lambda df: df.apply(pd.to_numeric, errors="coerce")
L, R = "Left", "Right"
def c(kind, stat, reg, side): return f"fs_{kind}_{stat}aparcstatsof{side}{reg}"
X["AS_HIPP"] = X[f"fs_vol_SubcorticalVolumeasegstatsof{L}Hippocampus"] + X[f"fs_vol_SubcorticalVolumeasegstatsof{R}Hippocampus"]
X["AS_AMYG"] = X[f"fs_vol_SubcorticalVolumeasegstatsof{L}Amygdala"] + X[f"fs_vol_SubcorticalVolumeasegstatsof{R}Amygdala"]
X["AS_ENT"] = (X[c("thk", "ThicknessAverage", "Entorhinal", L)] + X[c("thk", "ThicknessAverage", "Entorhinal", R)]) / 2
sig = [c("thk", "ThicknessAverage", reg, s) for reg in ("Entorhinal", "InferiorTemporal", "MiddleTemporal", "Fusiform") for s in (L, R)]
X["AS_SIG"] = X[sig].mean(axis=1, skipna=False)
X["AS_VENT"] = sum(X[f"fs_vol_SubcorticalVolumeasegstatsof{s}{v}"] for s in (L, R) for v in ("LateralVentricle", "InferiorLateralVentricle"))
COMPACT = ["AS_HIPP", "AS_AMYG", "AS_ENT", "AS_SIG", "AS_VENT"]

# ---------------- W-scores relative to amyloid-negative CN ----------------
ref = X[X["baseline_dx"].eq("CN") & X["A"].eq(0)]
cov = ["age", "sex_female"]
W = pd.DataFrame(index=X.index)
for col in fs_cols + COMPACT:
    r = ref[[col] + cov].apply(pd.to_numeric, errors="coerce").dropna()
    lr = LinearRegression().fit(r[cov].values, r[col].values)
    sd = np.std(r[col].values - lr.predict(r[cov].values), ddof=len(cov) + 1)
    xx = X[cov].apply(pd.to_numeric, errors="coerce")
    pred = np.full(len(X), np.nan); ok = xx.notna().all(axis=1).values
    pred[ok] = lr.predict(xx[ok].values)
    W["W_" + col] = (pd.to_numeric(X[col], errors="coerce").values - pred) / sd
X = pd.concat([X, W], axis=1)
W_all = ["W_" + c_ for c_ in fs_cols]; W_comp = ["W_" + c_ for c_ in COMPACT]
print("W-score reference: amyloid-negative CN n =", len(ref))

TASKS = {"amyloid": (X["baseline_dx"].eq("MCI") & X["A"].notna(), "A", ["A-", "A+"]),
         "trajectory": (X["traj"].notna(), "traj", ["stable", "slow", "fast"])}
SETS = {"clinical": demo + cog, "compact_raw": COMPACT, "compact_W": W_comp, "W_all": W_all,
        "clinical+compact_raw": demo + cog + COMPACT, "clinical+compact_W": demo + cog + W_comp,
        "clinical+W_all": demo + cog + W_all}
rows, drows, red_rows = [], [], []
for task, (mask, col, classes) in TASKS.items():
    sub = X[mask].reset_index(drop=True)
    y = sub[col].map({1.0: "A+", 0.0: "A-"}).values if col == "A" else sub[col].astype(str).values
    sp = repeated_kfold_first(y); P = {}
    for name, cols in SETS.items():
        P[name], _ = oof_predict(sub[cols], y, sp, "logistic_l2", classes, sub["RID"].values)
        a = auc(y, P[name], classes); lo, hi = boot_ci(y, P[name], classes)
        rows.append(dict(task=task, featureset=name, n=len(sub), n_features=len(cols), auc=round(a, 3),
                         ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
        print(rows[-1], flush=True)
    for name in ("clinical+compact_raw", "clinical+compact_W", "clinical+W_all"):
        d, lo, hi, p = paired_delta(y, P[name], P["clinical"], classes)
        drows.append(dict(task=task, comparison=f"{name} - clinical", delta=round(d, 3), ci_lo=round(lo, 3),
                          ci_hi=round(hi, 3), p=round(p, 4)))
    # (3c) unique atrophy information: residualise compact atrophy on clinical inside folds
    Xc = num(sub[demo + cog]); Xa = num(sub[COMPACT])
    Pres = np.full((len(y), len(classes)), np.nan)
    for tr, te in sp:
        med_c = Xc.iloc[tr].median(); med_a = Xa.iloc[tr].median()
        ctr, cte = Xc.iloc[tr].fillna(med_c).values, Xc.iloc[te].fillna(med_c).values
        atr, ate = Xa.iloc[tr].fillna(med_a).values, Xa.iloc[te].fillna(med_a).values
        reg = LinearRegression().fit(ctr, atr)
        rtr, rte = atr - reg.predict(ctr), ate - reg.predict(cte)
        mdl = clone(make_models()["logistic_l2"][0]).fit(rtr, y[tr]); cl = list(mdl.classes_)
        Pres[te] = mdl.predict_proba(rte)[:, [cl.index(k) for k in classes]]
    a = auc(y, Pres, classes); lo, hi = boot_ci(y, Pres, classes)
    rows.append(dict(task=task, featureset="residual_atrophy_only (unique to clinical)", n=len(sub),
                     n_features=len(COMPACT), auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
    print(rows[-1], flush=True)
    # (3b) correlation of FreeSurfer-only and clinical OOF risk (primary analysis files)
    if task == "amyloid":
        f1 = pd.read_csv(os.path.join(OUT, "oof_amyloid_freesurfer.csv")); f2 = pd.read_csv(os.path.join(OUT, "oof_amyloid_clinical.csv"))
        f3 = pd.read_csv(os.path.join(OUT, "oof_amyloid_cognition.csv"))
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        rho_c = stats.spearmanr(lg(f1["p_A+"]), lg(f2["p_A+"]))[0]; rho_g = stats.spearmanr(lg(f1["p_A+"]), lg(f3["p_A+"]))[0]
        red_rows.append(dict(analysis="Spearman rho, FreeSurfer-only vs clinical OOF logit (amyloid)", value=round(rho_c, 3)))
        red_rows.append(dict(analysis="Spearman rho, FreeSurfer-only vs cognition-only OOF logit (amyloid)", value=round(rho_g, 3)))
        # (3a) variance of atrophy explained
        for m_ in COMPACT:
            dd = num(sub[[m_] + demo + cog]).dropna()
            for lab_, prs in (("age+sex", ["age", "sex_female"]), ("age+sex+APOE", ["age", "sex_female", "APOE4"]),
                              ("age+sex+APOE+cognition", demo + cog)):
                r2 = LinearRegression().fit(dd[prs], dd[m_]).score(dd[prs], dd[m_])
                red_rows.append(dict(analysis=f"R2 of {m_} explained by {lab_} (MCI amyloid cohort, n={len(dd)})", value=round(r2, 3)))
        # (3d) in-sample likelihood-ratio test and odds ratios per SD
        dd = num(sub[demo + cog + COMPACT]).copy(); dd["y"] = (y == "A+").astype(int); dd = dd.dropna()
        z = (dd[demo + cog + COMPACT] - dd[demo + cog + COMPACT].mean()) / dd[demo + cog + COMPACT].std()
        m0 = sm.Logit(dd["y"], sm.add_constant(z[demo + cog])).fit(disp=0)
        m1 = sm.Logit(dd["y"], sm.add_constant(z[demo + cog + COMPACT])).fit(disp=0)
        lr_stat = 2 * (m1.llf - m0.llf); p_lr = stats.chi2.sf(lr_stat, len(COMPACT))
        red_rows.append(dict(analysis=f"LR test clinical vs clinical+compact atrophy (df=5, n={len(dd)}): chi2", value=round(lr_stat, 2)))
        red_rows.append(dict(analysis="LR test p", value=float(f"{p_lr:.3g}")))
        red_rows.append(dict(analysis="McFadden pseudo-R2 clinical", value=round(m0.prsquared, 3)))
        red_rows.append(dict(analysis="McFadden pseudo-R2 clinical+compact atrophy", value=round(m1.prsquared, 3)))
        for m_ in COMPACT + ["age", "APOE4", "ADAS13", "MMSE"]:
            red_rows.append(dict(analysis=f"OR per SD {m_} (clinical+compact model)",
                                 value=f"{np.exp(m1.params[m_]):.2f} (p={m1.pvalues[m_]:.3g})"))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "E_atrophy_models.csv"), index=False)
pd.DataFrame(drows).to_csv(os.path.join(OUT, "E_atrophy_delta.csv"), index=False)
pd.DataFrame(red_rows).to_csv(os.path.join(OUT, "E_redundancy.csv"), index=False)
print(pd.DataFrame(drows).to_string(index=False)); print(pd.DataFrame(red_rows).to_string(index=False))

# ---------------- (4) age-adjusted early atrophy rate by amyloid status ----------------
fsl = pd.read_csv(os.path.join(D, "freesurfer_fsx7.csv"), low_memory=False)
fsl["EXAMDATE"] = pd.to_datetime(fsl["EXAMDATE"], errors="coerce"); fsl = fsl.dropna(subset=["EXAMDATE"])
icv = pd.to_numeric(fsl["ST10CV"], errors="coerce")
fsl["HIPP"] = (pd.to_numeric(fsl["ST29SV"], errors="coerce") + pd.to_numeric(fsl["ST88SV"], errors="coerce")) / icv * 1000
fsl["VENT"] = (pd.to_numeric(fsl["ST37SV"], errors="coerce") + pd.to_numeric(fsl["ST96SV"], errors="coerce")) / icv * 1000
fsl["ENT"] = (pd.to_numeric(fsl["ST24TA"], errors="coerce") + pd.to_numeric(fsl["ST83TA"], errors="coerce")) / 2
recs = []
for rid, g in fsl.groupby("RID"):
    g = g.sort_values("EXAMDATE"); t0 = g["EXAMDATE"].min()
    yrs = (g["EXAMDATE"] - t0).dt.days.values / 365.25; win = (g["EXAMDATE"] - t0).dt.days.values <= 365
    rec = {"RID": rid}
    for m_ in ("HIPP", "VENT", "ENT"):
        v = pd.to_numeric(g[m_], errors="coerce").values; ok = win & np.isfinite(v)
        rec[f"sl_{m_}"] = np.polyfit(yrs[ok], v[ok], 1)[0] if (ok.sum() >= 2 and np.ptp(yrs[ok]) > 0.2) else np.nan
        rec[f"b0_{m_}"] = v[ok][0] if ok.any() else np.nan
    recs.append(rec)
sl = pd.DataFrame(recs)
am = X[X["baseline_dx"].eq("MCI") & X["A"].notna()][["RID", "A", "age", "sex_female"]].merge(sl, on="RID")
am = am.dropna(subset=["sl_HIPP", "sl_VENT", "sl_ENT"])
anc = []
for m_ in ("HIPP", "VENT", "ENT"):
    dd = am[["A", "age", "sex_female", f"b0_{m_}", f"sl_{m_}"]].apply(pd.to_numeric, errors="coerce").dropna()
    un = stats.ttest_ind(dd.loc[dd.A == 1, f"sl_{m_}"], dd.loc[dd.A == 0, f"sl_{m_}"], equal_var=False)
    fit = sm.OLS(dd[f"sl_{m_}"], sm.add_constant(dd[["A", "age", "sex_female", f"b0_{m_}"]])).fit(cov_type="HC3")
    ci = fit.conf_int().loc["A"]
    anc.append(dict(measure=m_, n=len(dd), n_Apos=int((dd.A == 1).sum()), n_Aneg=int((dd.A == 0).sum()),
                    Apos_mean=round(dd.loc[dd.A == 1, f"sl_{m_}"].mean(), 4), Aneg_mean=round(dd.loc[dd.A == 0, f"sl_{m_}"].mean(), 4),
                    unadjusted_p=float(f"{un.pvalue:.3g}"), adjusted_diff=round(fit.params["A"], 4),
                    adj_ci_lo=round(ci[0], 4), adj_ci_hi=round(ci[1], 4), adjusted_p=float(f"{fit.pvalues['A']:.3g}"),
                    age_coef=round(fit.params["age"], 5), age_p=float(f"{fit.pvalues['age']:.3g}")))
pd.DataFrame(anc).to_csv(os.path.join(OUT, "E_atrophy_rate_ancova.csv"), index=False)
print(pd.DataFrame(anc).to_string(index=False))
print("E done")
