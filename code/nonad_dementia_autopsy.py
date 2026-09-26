"""nonad_dementia_autopsy.py -- amyloid-negative (non-AD) dementia and autopsy validation.

(1) Non-AD dementia. ADNI enrols clinically diagnosed AD-type dementia only; clinician-
    recorded "dementia due to other etiology" at baseline is too rare to model, so amyloid-
    negative baseline dementia (biomarker-defined suspected non-AD dementia) is used as the
    non-AD group. Analyses (pre-specified logistic model, stratified 5-fold, seed 42):
      * four-class  CN / A-MCI / A+MCI / A- (non-AD) dementia   (reviewer's request)
      * five-class  CN / A-MCI / A+MCI / A+ (AD) dementia / A- (non-AD) dementia
      * binary      A+ (AD) dementia vs A- (non-AD) dementia
(2) Autopsy layer. (a) concordance of the in-vivo amyloid label with neuropathology
    (CERAD neuritic plaques moderate/frequent; NIA-AA ADNC intermediate/high);
    (b) discrimination of neuropathological amyloid by the clinical amyloid model in
    autopsied baseline-MCI participants, using out-of-fold predictions for those in the
    modelling cohort and a model trained on the full modelling cohort for those without an
    in-vivo label (never trained on the evaluated participant).
"""
from revision_common import *
from revision_common import _bin_auc
from scipy import stats
from sklearn.metrics import cohen_kappa_score

X, fam = load_all()
FS = feature_sets(fam)
SETS = ["demo_apoe", "cognition", "freesurfer", "clinical", "clinical+fs"]


def lvl(r, five):
    dx = r["baseline_dx"]
    if dx == "CN": return "CN"
    if dx == "MCI":
        return "A-MCI" if r["A"] == 0 else ("A+MCI" if r["A"] == 1 else np.nan)
    if dx == "AD":
        if r["A"] == 0: return "A-Dem"
        if r["A"] == 1 and five: return "A+Dem"
    return np.nan


rows, per_class, deltas = [], [], []
X["lvl4"] = X.apply(lambda r: lvl(r, False), axis=1)
X["lvl5"] = X.apply(lambda r: lvl(r, True), axis=1)
X["dem_amy"] = np.where(X["baseline_dx"].eq("AD") & X["A"].notna(), np.where(X["A"] == 1, "A+Dem", "A-Dem"), None)
TASKS = {"fourclass_nonAD": ("lvl4", ["CN", "A-MCI", "A+MCI", "A-Dem"]),
         "fiveclass": ("lvl5", ["CN", "A-MCI", "A+MCI", "A+Dem", "A-Dem"]),
         "AD_vs_nonAD_dementia": ("dem_amy", ["A-Dem", "A+Dem"])}
for task, (col, classes) in TASKS.items():
    sub = X[X[col].notna()].reset_index(drop=True); y = sub[col].astype(str).values
    sp = repeated_kfold_first(y); P = {}
    for fsname in SETS:
        cols = [c for c in FS[fsname] if c in sub.columns]
        P[fsname], _ = oof_predict(sub[cols], y, sp, "logistic_l2", classes, sub["RID"].values)
        a = auc(y, P[fsname], classes); lo, hi = boot_ci(y, P[fsname], classes)
        rows.append(dict(task=task, featureset=fsname, n=len(sub),
                         counts="/".join(f"{c}:{int((y == c).sum())}" for c in classes),
                         auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3), bacc=round(bacc(y, P[fsname], classes), 3)))
        if len(classes) > 2:
            for k, c in enumerate(classes):
                per_class.append(dict(task=task, featureset=fsname, cls=c, n=int((y == c).sum()),
                                      auc_ovr=round(_bin_auc(y == c, P[fsname][:, k]), 3)))
        print(rows[-1], flush=True)
    d, lo, hi, p = paired_delta(y, P["clinical"], P["clinical+fs"], classes)
    deltas.append(dict(task=task, comparison="clinical - (clinical+FS)", delta=round(d, 3),
                       ci_lo=round(lo, 3), ci_hi=round(hi, 3), p=round(p, 4)))
    d, lo, hi, p = paired_delta(y, P["clinical"], P["freesurfer"], classes)
    deltas.append(dict(task=task, comparison="clinical - FreeSurfer", delta=round(d, 3),
                       ci_lo=round(lo, 3), ci_hi=round(hi, 3), p=round(p, 4)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "D_nonAD_dementia.csv"), index=False)
pd.DataFrame(per_class).to_csv(os.path.join(OUT, "D_nonAD_per_class.csv"), index=False)
pd.DataFrame(deltas).to_csv(os.path.join(OUT, "D_nonAD_delta.csv"), index=False)
print(pd.DataFrame(deltas).to_string(index=False))

# characteristics of amyloid-positive vs amyloid-negative dementia
dem = X[X["dem_amy"].notna()].copy()
dem["HIPP"] = (dem["fs_vol_SubcorticalVolumeasegstatsofLeftHippocampus"] +
               dem["fs_vol_SubcorticalVolumeasegstatsofRightHippocampus"]) * 1000
chars = []
for c, lab in [("age", "Age, years"), ("sex_female", "Female"), ("PTEDUCAT", "Education, years"), ("APOE4", "APOE e4 carrier"),
               ("MMSE", "MMSE"), ("CDRSB", "CDR-SB"), ("ADAS13", "ADAS-Cog13"), ("FAQ", "FAQ"),
               ("HIPP", "Hippocampal volume / ICV (x1000)")]:
    a = pd.to_numeric(dem.loc[dem.dem_amy == "A+Dem", c], errors="coerce")
    b = pd.to_numeric(dem.loc[dem.dem_amy == "A-Dem", c], errors="coerce")
    if c in ("sex_female", "APOE4"):
        ta, tb = (a >= 1).sum(), (b >= 1).sum()
        p = stats.chi2_contingency([[ta, a.notna().sum() - ta], [tb, b.notna().sum() - tb]])[1]
        chars.append((lab, f"{100 * ta / a.notna().sum():.0f}%", f"{100 * tb / b.notna().sum():.0f}%", f"{p:.3g}"))
    else:
        p = stats.ttest_ind(a, b, nan_policy="omit", equal_var=False).pvalue
        chars.append((lab, f"{a.mean():.1f} ± {a.std():.1f}", f"{b.mean():.1f} ± {b.std():.1f}", f"{p:.3g}"))
pd.DataFrame(chars, columns=["characteristic", "A+ dementia", "A- dementia", "p"]).to_csv(
    os.path.join(OUT, "D_dementia_characteristics.csv"), index=False)
print(pd.DataFrame(chars).to_string(index=False))

# ---------------- autopsy ----------------
npth = pd.read_csv(os.path.join(D, "neuropath.csv"))
Z = X.merge(npth, on="RID", how="inner")
Z["NP_plaque_pos"] = np.where(Z["NPNEUR"].isna(), np.nan, (Z["NPNEUR"] >= 2).astype(float))
Z["NP_ADNC_pos"] = np.where(Z["NPADNC"].isna(), np.nan, (Z["NPADNC"] >= 2).astype(float))
Z["yrs_label_to_death"] = Z["NPDODYR"] - pd.to_datetime(
    np.where(Z["A_source"] == "PET", Z["PET_DATE"], Z["CSF_DATE"])).year
conc_rows = []
for ref in ("NP_plaque_pos", "NP_ADNC_pos"):
    k = Z["A"].notna() & Z[ref].notna()
    a, r = Z.loc[k, "A"].values, Z.loc[k, ref].values
    tp = int(((a == 1) & (r == 1)).sum()); tn = int(((a == 0) & (r == 0)).sum())
    fp = int(((a == 1) & (r == 0)).sum()); fn = int(((a == 0) & (r == 1)).sum())
    conc_rows.append(dict(reference=ref, n=int(k.sum()), agreement=round((tp + tn) / k.sum(), 3),
                          sensitivity=round(tp / (tp + fn), 3), specificity=round(tn / (tn + fp), 3),
                          kappa=round(cohen_kappa_score(a, r), 3), tp=tp, tn=tn, fp=fp, fn=fn,
                          median_years_label_to_death=float(Z.loc[k, "yrs_label_to_death"].median())))
pd.DataFrame(conc_rows).to_csv(os.path.join(OUT, "D_autopsy_label_concordance.csv"), index=False)
print(pd.DataFrame(conc_rows).to_string(index=False))

# model predictions for autopsied baseline-MCI
M = X[X["baseline_dx"].eq("MCI") & X["A"].notna()].reset_index(drop=True)
yM = np.where(M["A"] == 1, "A+", "A-")
aut_rows = []
for fsname in ("clinical", "freesurfer", "clinical+fs"):
    oof = pd.read_csv(os.path.join(OUT, f"oof_amyloid_{fsname}.csv"))[["RID", "p_A+"]]
    cols = [c for c in FS[fsname] if c in M.columns]
    est = make_models()["logistic_l2"][0]
    full = clone(est).fit(M[cols].apply(pd.to_numeric, errors="coerce").values, yM)
    ZM = Z[Z["baseline_dx"].eq("MCI")].copy()
    extra = ZM[~ZM["RID"].isin(oof["RID"])]
    pe = full.predict_proba(extra[cols].apply(pd.to_numeric, errors="coerce").values)[:, list(full.classes_).index("A+")]
    pr = pd.concat([oof, pd.DataFrame({"RID": extra["RID"].values, "p_A+": pe})])
    ZM = ZM.merge(pr, on="RID", how="left")
    for ref in ("NP_plaque_pos", "NP_ADNC_pos"):
        k = ZM[ref].notna() & ZM["p_A+"].notna()
        yy = np.where(ZM.loc[k, ref] == 1, "pos", "neg"); PP = np.c_[1 - ZM.loc[k, "p_A+"], ZM.loc[k, "p_A+"]]
        a = auc(yy, PP, ["neg", "pos"]); lo, hi = boot_ci(yy, PP, ["neg", "pos"])
        aut_rows.append(dict(model=fsname, reference=ref, n=int(k.sum()), n_pos=int((yy == "pos").sum()),
                             n_neg=int((yy == "neg").sum()), auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                             n_oof=int(ZM.loc[k, "RID"].isin(oof["RID"]).sum())))
    if fsname == "clinical":
        kk = ZM["A"].notna() & ZM["NP_plaque_pos"].notna()
        print("in-vivo label vs plaques among autopsied MCI with label: agreement",
              round((ZM.loc[kk, "A"] == ZM.loc[kk, "NP_plaque_pos"]).mean(), 3), "n", int(kk.sum()))
        print("years baseline->death (MCI autopsy):", (ZM["NPDODYR"] - ZM["baseline_date"].dt.year).describe().round(1).to_dict())
pd.DataFrame(aut_rows).to_csv(os.path.join(OUT, "D_autopsy_model.csv"), index=False)
print(pd.DataFrame(aut_rows).to_string(index=False))
print("D done")

# paired comparison of models within the autopsy subset
pz = {}
for fsname in ("clinical", "freesurfer", "clinical+fs"):
    oof = pd.read_csv(os.path.join(OUT, f"oof_amyloid_{fsname}.csv"))[["RID", "p_A+"]]
    cols = [c for c in FS[fsname] if c in M.columns]
    full = clone(make_models()["logistic_l2"][0]).fit(M[cols].apply(pd.to_numeric, errors="coerce").values, yM)
    ZM = Z[Z["baseline_dx"].eq("MCI")].copy(); extra = ZM[~ZM["RID"].isin(oof["RID"])]
    pe = full.predict_proba(extra[cols].apply(pd.to_numeric, errors="coerce").values)[:, list(full.classes_).index("A+")]
    pz[fsname] = ZM[["RID", "NP_plaque_pos", "NP_ADNC_pos"]].merge(
        pd.concat([oof, pd.DataFrame({"RID": extra["RID"].values, "p_A+": pe})]), on="RID", how="left")
pdr = []
for ref in ("NP_plaque_pos", "NP_ADNC_pos"):
    base = pz["clinical"]; k = base[ref].notna().values
    yy = np.where(base.loc[k, ref] == 1, "pos", "neg")
    Pc = np.c_[1 - base.loc[k, "p_A+"], base.loc[k, "p_A+"]]
    for other in ("clinical+fs", "freesurfer"):
        o = pz[other].set_index("RID").loc[base.loc[k, "RID"], "p_A+"].values
        d, lo, hi, p = paired_delta(yy, np.c_[1 - o, o], Pc, ["neg", "pos"])
        pdr.append(dict(reference=ref, comparison=f"{other} - clinical", delta=round(d, 3), ci_lo=round(lo, 3),
                        ci_hi=round(hi, 3), p=round(p, 4), n=int(k.sum())))
pd.DataFrame(pdr).to_csv(os.path.join(OUT, "D_autopsy_delta.csv"), index=False)
print(pd.DataFrame(pdr).to_string(index=False))
