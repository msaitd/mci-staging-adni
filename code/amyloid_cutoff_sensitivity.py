"""amyloid_cutoff_sensitivity.py -- amyloid cut-off / assay dependence.

Re-derives the amyloid label of baseline-MCI participants under alternative, published
PET and CSF definitions and re-runs the amyloid staging for each definition with the same
pre-specified logistic pipeline and the same partitioning rule (stratified 5-fold, seed 42).

PET (Centiloid, CL): 12 (Salvado 2019), 20 (primary; Amadoru 2020 / Royse 2021),
24.1 (Navitsky 2018), 30 (Salvado 2019); tracer-specific UC Berkeley SUVR status
(florbetapir 1.11, florbetaben 1.08; Royse 2021).
CSF (Elecsys): Abeta42 < 880 / 980 / 1100 pg/mL and p-tau181/Abeta42 > 0.025 / 0.028
(Hansson 2018; Shaw 2018). Also: CSF-preferred, PET-only, CSF-only, PET-CSF concordant
only, equivocal PET zone (12-30 CL) removed, and a time-matched label (measurement
closest to the clinical baseline within +/-12 months).
"""
from revision_common import *

X, fam = load_all()
FS = feature_sets(fam)
M = X[X["baseline_dx"].eq("MCI")].reset_index(drop=True)
pet, csf = M["AMYPET_CENTILOID"], M["CSF_ABETA42"]
ratio = M["CSF_PTAU181"] / M["CSF_AB42"]
ucb = M["PET_STATUS_UCB"]
nan = np.nan


def comb(first, second):
    return np.where(pd.notna(first), first, second)


def pos(v, thr, below=False):
    v = pd.Series(v)
    out = np.where(v.isna(), nan, ((v < thr) if below else (v >= thr)).astype(float))
    return out


dt_pet = (M["PET_DATE"] - M["baseline_date"]).dt.days.abs() / 30.44
dt_csf = (M["CSF_DATE"] - M["baseline_date"]).dt.days.abs() / 30.44
use_pet = pet.notna() & (dt_pet <= 12) & (~(csf.notna() & (dt_csf <= 12)) | (dt_pet <= dt_csf))
use_csf = csf.notna() & (dt_csf <= 12) & ~use_pet
time_matched = np.where(use_pet, pos(pet, 20), np.where(use_csf, pos(csf, 980, True), nan))

conc = np.where(pet.notna() & csf.notna() & (pos(pet, 20) == pos(csf, 980, True)), pos(pet, 20), nan)
equiv = np.where(pet.notna() & (pet >= 12) & (pet < 30), nan, M["A"].values)

DEFS = {
    "Primary: PET >=20 CL, else CSF Abeta42 <980": M["A"].values,
    "PET >=12 CL, else CSF <980": comb(pos(pet, 12), pos(csf, 980, True)),
    "PET >=24.1 CL, else CSF <980": comb(pos(pet, 24.1), pos(csf, 980, True)),
    "PET >=30 CL, else CSF <980": comb(pos(pet, 30), pos(csf, 980, True)),
    "PET tracer-specific SUVR (UC Berkeley), else CSF <980": comb(np.where(pet.notna(), ucb, nan), pos(csf, 980, True)),
    "PET >=20 CL, else CSF Abeta42 <880": comb(pos(pet, 20), pos(csf, 880, True)),
    "PET >=20 CL, else CSF Abeta42 <1100": comb(pos(pet, 20), pos(csf, 1100, True)),
    "CSF-preferred: Abeta42 <980, else PET >=20": comb(pos(csf, 980, True), pos(pet, 20)),
    "CSF-preferred: p-tau181/Abeta42 >0.025, else PET >=20": comb(pos(ratio, 0.025), pos(pet, 20)),
    "CSF-preferred: p-tau181/Abeta42 >0.028, else PET >=20": comb(pos(ratio, 0.028), pos(pet, 20)),
    "PET only (>=20 CL)": pos(pet, 20),
    "CSF only (Abeta42 <980)": pos(csf, 980, True),
    "CSF only (p-tau181/Abeta42 >0.025)": pos(ratio, 0.025),
    "PET-CSF concordant only": conc,
    "Equivocal PET zone (12-30 CL) excluded": equiv,
    "Time-matched (closest measurement within +/-12 months)": time_matched,
}
prim = M["A"].values
rows = []
for name, lab in DEFS.items():
    lab = np.asarray(lab, float); keep = ~np.isnan(lab)
    sub = M[keep].reset_index(drop=True); yl = lab[keep]
    y = np.where(yl == 1, "A+", "A-"); classes = ["A-", "A+"]
    both = keep & ~np.isnan(prim)
    recl = float((lab[both] != prim[both]).mean()) if both.any() else nan
    sp = repeated_kfold_first(y)
    P = {}
    for fsname in ("demo_apoe", "freesurfer", "clinical", "clinical+fs"):
        cols = [c for c in FS[fsname] if c in sub.columns]
        P[fsname], _ = oof_predict(sub[cols], y, sp, "logistic_l2", classes, sub["RID"].values)
    a = auc(y, P["clinical"], classes); lo, hi = boot_ci(y, P["clinical"], classes)
    d, dlo, dhi, dp = paired_delta(y, P["clinical"], P["clinical+fs"], classes)
    rows.append(dict(definition=name, n=int(keep.sum()), n_Apos=int((yl == 1).sum()), n_Aneg=int((yl == 0).sum()),
                     pct_reclassified_vs_primary=round(100 * recl, 1) if recl == recl else "",
                     auc_demo_apoe=round(auc(y, P["demo_apoe"], classes), 3),
                     auc_freesurfer=round(auc(y, P["freesurfer"], classes), 3),
                     auc_clinical=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                     auc_clinical_fs=round(auc(y, P["clinical+fs"], classes), 3),
                     delta_clin_minus_clinfs=round(d, 3), delta_lo=round(dlo, 3), delta_hi=round(dhi, 3),
                     delta_p=round(dp, 4)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "C_cutoff_sensitivity.csv"), index=False)
# time between label-defining measurement and baseline (primary label)
mm = M[M["A"].notna()]
desc = mm["A_dt_months"].describe(percentiles=[.25, .5, .75, .9]).round(2)
desc["n_within_12m"] = int((mm["A_dt_months"].abs() <= 12).sum())
desc["n_beyond_12m"] = int((mm["A_dt_months"].abs() > 12).sum())
desc.to_csv(os.path.join(OUT, "C_label_timing.csv")); print(desc)
print(mm.groupby("A_source")["A_dt_months"].describe().round(2))
print("C done")
