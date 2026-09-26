"""OASIS-3 external validation of the frozen ADNI 'portable' models (Paper B, revision).

Run locally by the authorised OASIS data user:
    python oasis3_external_validation.py

Inputs : OASIS-3 data files under DATA_ROOT (as downloaded from NITRC-IR, 0AS_data_files)
         frozen_adni_models.json (ADNI model parameters only; same folder as this script)
Outputs: ./oasis_results/  -- AGGREGATE results only (no participant IDs, no participant-level values)

Pre-specified analyses
  Aim 1  frozen ADNI amyloid model in OASIS-3 MCI (Centiloid >= 20; sensitivity 12/24.1/30 CL, PVC Centiloid,
         tracer subgroups, QC-passed only, alternative MCI definitions): AUC, calibration, Brier, decision curve.
  Aim 2  incremental value of FreeSurfer measures, refitted within OASIS-3 (5-fold CV, paired bootstrap):
         portable vs portable + compact AD-signature (5) vs portable + full FreeSurfer panel.
  Aim 3  frozen ADNI trajectory model (stable / slow / fast progression to dementia).
  Aim 4  frozen ADNI four-group model (CN / A-MCI / A+MCI / AD dementia).
"""
import os, re, sys, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# folder with the OASIS-3 tables downloaded from NITRC-IR (0AS_data_files); set OASIS_ROOT or edit this line
DATA_ROOT = os.environ.get("OASIS_ROOT", os.path.join(os.path.dirname(HERE), "oasis3_data"))
OUT = os.path.join(HERE, "oasis_results"); os.makedirs(OUT, exist_ok=True)
FROZEN = json.load(open(os.path.join(HERE, "frozen_adni_models.json"), encoding="utf-8"))
PORT = FROZEN["features"]
WINDOW = 365          # max days between the anchor (PET / MRI) and the clinical visit
N_BOOT = 2000
SEED = 42
LOGF = open(os.path.join(OUT, "oasis_log.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOGF.write(s + "\n"); LOGF.flush()


def find(pattern):
    hits = glob.glob(os.path.join(DATA_ROOT, "**", pattern), recursive=True)
    if not hits:
        sys.exit(f"File not found under {DATA_ROOT}: {pattern}")
    return hits[0]


def read(pattern, usecols=None):
    df = pd.read_csv(find(pattern), low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if usecols is not None:
        df = df[[c for c in usecols if c in df.columns]]
    return df


num = lambda s: pd.to_numeric(s, errors="coerce")


def days_from(x):
    m = re.search(r"_d(\d+)", str(x)); return float(m.group(1)) if m else np.nan


def subj_from(x):
    m = re.search(r"(OAS3\d{4})", str(x)); return m.group(1) if m else None


def profile(name, s, top=25):
    """Aggregate description of one variable (for checking codings)."""
    if s.dtype == object or s.nunique(dropna=True) <= 12:
        vc = s.astype(str).value_counts(dropna=False).head(top)
        log(f"  [{name}] value counts: " + "; ".join(f"{k}={v if v >= 5 else '<5'}" for k, v in vc.items()))
    else:
        v = num(s)
        log(f"  [{name}] n={v.notna().sum()} missing={v.isna().mean():.2%} min={v.min():.3g} "
            f"median={v.median():.3g} max={v.max():.3g}")


# ============================ statistics ============================
def bin_auc(y, p):
    y = np.asarray(y).astype(bool); p = np.asarray(p, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(p).rank().values
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_any(y, P, classes):
    y = np.asarray(y).astype(str)
    if len(classes) == 2:
        return bin_auc(y == classes[1], P[:, 1])
    return float(np.nanmean([bin_auc(y == c, P[:, k]) for k, c in enumerate(classes)]))


def boot(y, P, classes, fn=auc_any, n=N_BOOT, seed=1):
    y = np.asarray(y).astype(str); rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < len(classes):
            continue
        vals.append(fn(y[s], P[s], classes))
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def paired_delta(y, PA, PB, classes, n=N_BOOT, seed=2):
    """AUC(B) - AUC(A) with paired subject-level bootstrap."""
    y = np.asarray(y).astype(str); rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < len(classes):
            continue
        d.append(auc_any(y[s], PB[s], classes) - auc_any(y[s], PA[s], classes))
    d = np.array(d); est = auc_any(y, PB, classes) - auc_any(y, PA, classes)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return est, np.percentile(d, 2.5), np.percentile(d, 97.5), p


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def logistic_fit(X, y, offset=None, iters=100):
    """Newton-Raphson logistic regression (for calibration slope/intercept)."""
    X = np.column_stack([np.ones(len(y))] + ([X] if X is not None else []))
    off = np.zeros(len(y)) if offset is None else offset
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b + off; mu = 1 / (1 + np.exp(-eta)); W = mu * (1 - mu) + 1e-12
        step = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (y - mu))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def calibration(y, p):
    y = np.asarray(y, float); lp = logit(p)
    a, b = logistic_fit(lp, y)                        # slope model
    citl = logistic_fit(None, y, offset=lp)[0]        # calibration-in-the-large
    return dict(brier=float(np.mean((p - y) ** 2)), cal_slope=float(b), cal_intercept_citl=float(citl),
                mean_pred=float(p.mean()), observed=float(y.mean()))


def net_benefit(y, p, t):
    y = np.asarray(y, bool); n = len(y); pos = p >= t
    return (np.sum(pos & y) - np.sum(pos & ~y) * t / (1 - t)) / n


def frozen_proba(task, Xdf):
    m = FROZEN["models"][task]
    Xv = Xdf[PORT].apply(num).values.astype(float)
    Z = np.where(np.isnan(Xv), np.array(m["impute_median"]), Xv)
    Z = (Z - np.array(m["scaler_mean"])) / np.array(m["scaler_scale"])
    coef, b = np.array(m["coef"]), np.array(m["intercept"])
    if len(m["classes"]) == 2:
        p1 = 1 / (1 + np.exp(-(Z @ coef[0] + b[0]))); return np.column_stack([1 - p1, p1]), m["classes"]
    L = Z @ coef.T + b; L -= L.max(axis=1, keepdims=True); E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True), m["classes"]


def cv_proba(Xdf, y, classes, n_splits=5, seed=SEED):
    """Refit inside OASIS: same pipeline as ADNI, stratified 5-fold, out-of-fold probabilities."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    Xv = Xdf.apply(num).values.astype(float); y = np.asarray(y).astype(str)
    keep = ~np.all(np.isnan(Xv), axis=0); Xv = Xv[:, keep]
    P = np.full((len(y), len(classes)), np.nan)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Xv, y):
        m = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]).fit(Xv[tr], y[tr])
        cl = list(m.classes_); P[te] = m.predict_proba(Xv[te])[:, [cl.index(c) for c in classes]]
    return P


ROWS = []


def add(analysis, subset, model, y, P, classes, extra=None):
    y = np.asarray(y).astype(str); a = auc_any(y, P, classes); lo, hi = boot(y, P, classes)
    r = dict(analysis=analysis, subset=subset, model=model, n=len(y),
             class_counts="; ".join(f"{c}={int((y == c).sum())}" for c in classes),
             auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3))
    if len(classes) == 2:
        r.update({k: round(v, 3) for k, v in calibration((y == classes[1]).astype(int), P[:, 1]).items()})
    if extra:
        r.update(extra)
    ROWS.append(r); log(f"  {analysis} | {subset} | {model}: n={len(y)} AUC {a:.3f} ({lo:.3f}-{hi:.3f})")
    return r


# ============================ load and harmonise ============================
log("OASIS-3 external validation -- aggregate log (no participant-level data)")
log("DATA_ROOT:", DATA_ROOT)

demo = read("OASIS3_demographics.csv")
log("\n[demographics] rows:", len(demo))
for c in ("GENDER", "APOE", "EDUC", "AgeatEntry"):
    if c in demo: profile(c, demo[c])
demo["sid"] = demo["OASISID"].map(subj_from)
g = demo["GENDER"]
if num(g).notna().mean() > 0.5:
    demo["sex_female"] = num(g).map({1: 0.0, 2: 1.0})
else:
    demo["sex_female"] = g.astype(str).str.upper().str[0].map({"M": 0.0, "F": 1.0})
apo = num(demo["APOE"]).round().astype("Int64").astype(str)
demo["APOE4"] = np.where(apo.isin(["22", "23", "24", "33", "34", "44"]), apo.str.count("4"), np.nan)
demo["PTEDUCAT"] = num(demo["EDUC"])
demo["AgeatEntry"] = num(demo["AgeatEntry"])
log("  derived: female share", round(demo["sex_female"].mean(), 3), "| APOE4 counts",
    demo["APOE4"].value_counts(dropna=False).to_dict())
demo = demo.drop_duplicates("sid").set_index("sid")

b4 = read("OASIS3_UDSb4_cdr.csv")
log("\n[UDS B4 / CDR] rows:", len(b4))
for c in ("CDRTOT", "MMSE", "CDRSUM", "dx1"):
    if c in b4: profile(c, b4[c])
b4["sid"] = b4["OASISID"].map(subj_from); b4["day"] = num(b4["days_to_visit"])
b4["age_visit"] = num(b4["age at visit"]); b4["MMSE"] = num(b4["MMSE"])
b4["CDRSB"] = num(b4["CDRSUM"]); b4["CDRTOT"] = num(b4["CDRTOT"])
b4["dx1"] = b4["dx1"].astype(str)

d1 = read("OASIS3_UDSd1_diagnoses.csv")
log("\n[UDS D1 / diagnosis] rows:", len(d1))
for c in ("DEMENTED", "NORMCOG", "MCIAMEM", "PROBAD", "POSSAD", "alzdis"):
    if c in d1: profile(c, d1[c])
d1["sid"] = d1["OASISID"].map(subj_from); d1["day"] = num(d1["days_to_visit"])
flags = [c for c in ("DEMENTED", "NORMCOG", "MCIAMEM", "MCIAPLUS", "MCINON1", "MCINON2", "PROBAD", "POSSAD", "alzdis") if c in d1]
for c in flags:
    d1[c] = num(d1[c])
d1 = d1.drop_duplicates(["sid", "day"])[["sid", "day"] + flags]
vis = b4.merge(d1, on=["sid", "day"], how="left")
DEM_TXT = re.compile(r"dem|\bdat\b|dlbd|alzheim|ftd|pick|lewy", re.I)
AD_TXT = re.compile(r"\bad\b|\bdat\b|alzheim", re.I)
vis["dem_txt"] = vis["dx1"].map(lambda s: bool(DEM_TXT.search(s)) if s not in ("nan", "") else np.nan)
vis["demented"] = np.where(vis["DEMENTED"].notna(), vis["DEMENTED"] == 1, vis["dem_txt"] == True)
mci_uds = vis[[c for c in ("MCIAMEM", "MCIAPLUS", "MCINON1", "MCINON2") if c in vis]].eq(1).any(axis=1)
vis["mci_uds"] = mci_uds
log("\n  dx1 among CDR 0.5 visits (top categories):")
vc = vis.loc[vis["CDRTOT"] == 0.5, "dx1"].value_counts().head(30)
for k, v in vc.items():
    log(f"    {k}: {v if v >= 5 else '<5'}  (dementia-text={bool(DEM_TXT.search(k))})")

# MCI definitions at a clinical visit.
# The Knight ADRC routinely assigns a clinical dementia label (e.g., "AD Dementia", "uncertain dementia") at CDR 0.5,
# so the clinician label is not comparable with ADNI's MCI category. The primary definition therefore uses ADNI's
# operational MCI criteria: CDR global 0.5 and MMSE >= 24 (MMSE from the nearest visit with an MMSE, within the window).
NONAD_TXT = re.compile(r"dlbd|vascular|frontotemporal|ftd|/pd|pd,|parkinson|non ad|non-ad|non dat|huntington|prion", re.I)
vis["nonad_dx"] = vis["dx1"].map(lambda s: bool(NONAD_TXT.search(s)))
vis["cdr05"] = vis["CDRTOT"] == 0.5
vis["cdr05_nononad"] = vis["cdr05"] & ~vis["nonad_dx"]
vis["cdr05_notdem"] = vis["cdr05"] & ~vis["demented"].astype(bool)
MCI_DEFS = {  # label: (visit flag, minimum MMSE or None)
    "primary: CDR 0.5 and MMSE>=24 (ADNI-harmonised)": ("cdr05", 24),
    "sensitivity: CDR 0.5, MMSE>=24, clinician non-AD diagnoses excluded": ("cdr05_nononad", 24),
    "sensitivity: CDR 0.5, MMSE>=24, clinician label 'not demented'": ("cdr05_notdem", 24),
    "sensitivity: CDR 0.5, any MMSE": ("cdr05", None),
}
PRIMARY = list(MCI_DEFS)[0]
log("\n  CDR 0.5 visits: %d; with a clinician non-AD diagnosis: %d; labelled 'not demented': %d" % (
    vis["cdr05"].sum(), (vis["cdr05"] & vis["nonad_dx"]).sum(), vis["cdr05_notdem"].sum()))

cl = read("OASIS3_amyloid_centiloid.csv")
log("\n[Centiloid] rows:", len(cl))
for c in ("tracer",):
    profile(c, cl[c])
cl["sid"] = cl["subject_id"].map(subj_from); cl["day"] = cl["oasis_session_id"].map(days_from)
cl["CL"] = num(cl["Centiloid_fSUVR_TOT_CORTMEAN"]); cl["CL_pvc"] = num(cl.get("Centiloid_fSUVR_rsf_TOT_CORTMEAN"))
cl["tracer"] = cl["tracer"].astype(str).str.upper()
profile("Centiloid_fSUVR_TOT_CORTMEAN", cl["CL"]); profile("Centiloid_fSUVR_rsf_TOT_CORTMEAN", cl["CL_pvc"])
cl = cl[cl["CL"].notna() & cl["day"].notna() & cl["sid"].notna()].copy()

# PUP quality control status (joined on subject, day, tracer)
cl["qc_pass"] = np.nan
try:
    pup = read("OASIS3_PUP.csv", usecols=["PUP_PUPTIMECOURSEDATA ID", "tracer", "PET TC QC Status", "FreeSurfer QC Status"])
    log("\n[PUP] rows:", len(pup)); profile("PET TC QC Status", pup["PET TC QC Status"]); profile("FreeSurfer QC Status", pup["FreeSurfer QC Status"])
    pup["sid"] = pup["PUP_PUPTIMECOURSEDATA ID"].map(subj_from); pup["day"] = pup["PUP_PUPTIMECOURSEDATA ID"].map(days_from)
    pup["tracer"] = pup["tracer"].astype(str).str.upper()
    q = (pup["PET TC QC Status"].astype(str) + " " + pup["FreeSurfer QC Status"].astype(str)).str.lower()
    pup["qc_pass"] = (~q.str.contains("fail|reject|unusable")).astype(float)
    cl = cl.drop(columns="qc_pass").merge(pup.drop_duplicates(["sid", "day", "tracer"])[["sid", "day", "tracer", "qc_pass"]],
                                          on=["sid", "day", "tracer"], how="left")
    log("  Centiloid sessions matched to PUP QC:", int(cl["qc_pass"].notna().sum()), "of", len(cl),
        "| QC-passed:", int((cl["qc_pass"] == 1).sum()))
except SystemExit:
    log("  OASIS3_PUP.csv not found -- QC sensitivity analysis skipped")


def nearest(anchor, table, cols, require=None, window=WINDOW):
    t = table[["sid", "day"] + cols].copy()
    if require:
        t = t[t[require].notna()]
    m = anchor[["aid", "sid", "day"]].merge(t, on="sid", suffixes=("", "_t"))
    m["gap"] = (m["day_t"] - m["day"]).abs()
    m = m[m["gap"] <= window].sort_values(["aid", "gap"]).drop_duplicates("aid")
    return m.set_index("aid")


# ============================ Aim 1: amyloid cohort ============================
def amyloid_cohort(defn, tracers=None):
    mci_col, mmin = defn
    a = cl if tracers is None else cl[cl["tracer"].isin(tracers)]
    a = a.reset_index(drop=True).copy(); a["aid"] = np.arange(len(a))
    v = nearest(a, vis, ["age_visit", "CDRSB", "CDRTOT", mci_col])
    a = a.join(v[["age_visit", "CDRSB", "CDRTOT", mci_col, "gap"]], on="aid")
    mm = nearest(a, vis, ["MMSE"], require="MMSE")
    a = a.join(mm[["MMSE"]], on="aid")
    ok = a[mci_col] == True
    if mmin is not None:
        ok &= a["MMSE"] >= mmin
    a = a[ok].sort_values(["sid", "day"]).drop_duplicates("sid")   # earliest qualifying PET
    a = a.join(demo[["sex_female", "PTEDUCAT", "APOE4", "AgeatEntry"]], on="sid")
    a["age"] = a["age_visit"].fillna(a["AgeatEntry"] + a["day"] / 365.25)
    return a.reset_index(drop=True)


log("\n================ Aim 1: frozen ADNI amyloid model ================")
log("ADNI reference (cross-validated, same predictors):", FROZEN["models"]["amyloid"]["adni_cv_auc"],
    FROZEN["models"]["amyloid"]["adni_cv_auc_ci"])
log(f"Flow: Centiloid sessions {len(cl)} from {cl['sid'].nunique()} participants")
# Pre-specified in the OASIS data request: primary analyses use PiB; AV45 (florbetapir) results may only be
# published after review by Avid Radiopharmaceuticals (OASIS Data Use Agreement, term 8).
PRIMARY_TRACERS = ["PIB"] if (cl["tracer"] == "PIB").any() else None
log("Primary tracer(s):", PRIMARY_TRACERS or "all (PIB not found)")
cohorts = {}
for label, defn in MCI_DEFS.items():
    A = amyloid_cohort(defn, PRIMARY_TRACERS); cohorts[label] = A
    log(f"  {label}: {len(A)} participants; PET-to-clinical gap median {A['gap'].median():.0f} days "
        f"(IQR {A['gap'].quantile(.25):.0f}-{A['gap'].quantile(.75):.0f}); tracers {A['tracer'].value_counts().to_dict()}")

A = cohorts[PRIMARY]
log("\nPredictor missingness in the primary cohort (imputed with ADNI training medians):")
for c in PORT:
    log(f"  {c}: {A[c].isna().mean():.1%} missing")
CLS = ["A-", "A+"]   # column 1 = probability of amyloid positivity


def amy_proba(df):
    P, cls = frozen_proba("amyloid", df)
    pA = P[:, cls.index("A+")]
    return np.column_stack([1 - pA, pA])


def amyloid_eval(Adf, thr, subset, cl_col="CL"):
    d = Adf[Adf[cl_col].notna()]
    y = np.where(d[cl_col] >= thr, "A+", "A-")
    P = amy_proba(d)
    return add("Aim1 amyloid (frozen ADNI portable model)", subset, "portable", y, P, CLS), d, y, P


# primary + sensitivity analyses
_, dA, yA, PA = amyloid_eval(A, 20, "primary cohort, Centiloid >= 20")
for thr in (12, 24.1, 30):
    amyloid_eval(A, thr, f"primary cohort, Centiloid >= {thr}")
amyloid_eval(A, 20, "primary cohort, PVC (rsf) Centiloid >= 20", cl_col="CL_pvc")
if PRIMARY_TRACERS:
    A_all = amyloid_cohort(MCI_DEFS[PRIMARY])
    log(f"  all tracers: {len(A_all)} participants; tracers {A_all['tracer'].value_counts().to_dict()}")
    note = " [includes AV45: Avid review required before publication]"
    amyloid_eval(A_all, 20, "all tracers (PIB + AV45), Centiloid >= 20" + note)
    A_av = amyloid_cohort(MCI_DEFS[PRIMARY], ["AV45"])
    if len(A_av) >= 30:
        amyloid_eval(A_av, 20, "AV45 only, Centiloid >= 20" + note)
if A["qc_pass"].notna().any():
    amyloid_eval(A[A["qc_pass"] == 1], 20, "PUP QC passed only, Centiloid >= 20")
amyloid_eval(A[A["gap"] <= 90], 20, "clinical visit within 90 days of PET")
for label, Ad in cohorts.items():
    if not label.startswith("primary") and len(Ad) >= 30:
        amyloid_eval(Ad, 20, label)
for lab_, k in (("APOE e4 non-carriers", A["APOE4"] == 0), ("APOE e4 carriers", A["APOE4"] >= 1)):
    if k.sum() >= 30:
        amyloid_eval(A[k], 20, lab_)
if len(A) >= 90:
    tert = pd.qcut(A["age"], 3, labels=False, duplicates="drop")
    for t in sorted(tert.dropna().unique()):
        k = tert == t
        amyloid_eval(A[k], 20, f"age tertile {int(t) + 1} ({A.loc[k, 'age'].min():.0f}-{A.loc[k, 'age'].max():.0f} y)")

# refit inside OASIS (how much is lost by transport)
Pref = cv_proba(dA[PORT], yA, CLS)
add("Aim1 amyloid (refitted in OASIS, 5-fold CV)", "primary cohort, Centiloid >= 20", "portable (refit)", yA, Pref, CLS)

# recalibration-in-the-large, operating points, decision curve (primary)
p = PA[:, 1]
yb = (yA == "A+").astype(int)
cal = calibration(yb, p); p_rec = 1 / (1 + np.exp(-(logit(p) + cal["cal_intercept_citl"])))
cal_r = calibration(yb, p_rec)
ops = []
for name, t in (("ADNI 90%-sensitivity threshold (0.25)", 0.25),
                ("OASIS 90%-sensitivity threshold", float(np.quantile(p[yb == 1], 0.10)))):
    pos = p >= t
    tp, fp, fn, tn = (pos & (yb == 1)).sum(), (pos & (yb == 0)).sum(), (~pos & (yb == 1)).sum(), (~pos & (yb == 0)).sum()
    ops.append(dict(operating_point=name, threshold=round(t, 3), sensitivity=round(tp / (tp + fn), 3),
                    specificity=round(tn / (tn + fp), 3), ppv=round(tp / max(tp + fp, 1), 3),
                    npv=round(tn / max(tn + fn, 1), 3), fraction_referred=round(pos.mean(), 3)))
pd.DataFrame(ops).to_csv(os.path.join(OUT, "oasis_operating_points.csv"), index=False)
pd.DataFrame([dict(model="frozen", **cal), dict(model="intercept-recalibrated", **cal_r)]).round(3).to_csv(
    os.path.join(OUT, "oasis_calibration_summary.csv"), index=False)
dec = pd.qcut(p, 10, labels=False, duplicates="drop")
pd.DataFrame({"decile": dec, "p": p, "y": yb}).groupby("decile").agg(n=("y", "size"), mean_pred=("p", "mean"),
                                                                     observed=("y", "mean")).round(3).to_csv(
    os.path.join(OUT, "oasis_calibration_deciles.csv"))
pd.DataFrame([dict(threshold=t, model=round(net_benefit(yb, p, t), 4),
                   model_recalibrated=round(net_benefit(yb, p_rec, t), 4),
                   test_all=round(yb.mean() - (1 - yb.mean()) * t / (1 - t), 4), test_none=0.0)
              for t in (0.1, 0.2, 0.3, 0.4, 0.5)]).to_csv(os.path.join(OUT, "oasis_decision_curve.csv"), index=False)
log(f"  calibration (frozen): {cal} | after intercept recalibration: Brier {cal_r['brier']:.3f}")

# Table 1
try:
    from scipy import stats
except ImportError:
    stats = None
t1 = []
for c, lab, kind in (("age", "Age, years", "c"), ("sex_female", "Female", "b"), ("PTEDUCAT", "Education, years", "c"),
                     ("APOE4", "APOE e4 carrier", "b"), ("MMSE", "MMSE", "c"), ("CDRSB", "CDR-SB", "c"),
                     ("CL", "Centiloid", "c")):
    g1, g0 = dA.loc[yA == "A+", c], dA.loc[yA == "A-", c]
    if kind == "c":
        pv = stats.ttest_ind(g1.dropna(), g0.dropna(), equal_var=False).pvalue if stats else np.nan
        t1.append(dict(characteristic=lab, A_pos=f"{g1.mean():.1f} ± {g1.std():.1f}", A_neg=f"{g0.mean():.1f} ± {g0.std():.1f}", p=pv))
    else:
        b1, b0 = (g1 >= 1).where(g1.notna()), (g0 >= 1).where(g0.notna())
        tab = [[int(b1.sum()), int(b1.notna().sum() - b1.sum())], [int(b0.sum()), int(b0.notna().sum() - b0.sum())]]
        pv = stats.chi2_contingency(tab)[1] if stats else np.nan
        t1.append(dict(characteristic=lab, A_pos=f"{100 * b1.mean():.0f}%", A_neg=f"{100 * b0.mean():.0f}%", p=pv))
t1.append(dict(characteristic="n", A_pos=int((yA == "A+").sum()), A_neg=int((yA == "A-").sum()), p=np.nan))
for tr in sorted(dA["tracer"].unique()):
    t1.append(dict(characteristic=f"Tracer {tr}", A_pos=int(((dA["tracer"] == tr) & (yA == "A+")).sum()),
                   A_neg=int(((dA["tracer"] == tr) & (yA == "A-")).sum()), p=np.nan))
pd.DataFrame(t1).to_csv(os.path.join(OUT, "oasis_table1.csv"), index=False)

# ============================ Aim 2: FreeSurfer, refitted in OASIS ============================
log("\n================ Aim 2: incremental value of FreeSurfer (refitted in OASIS) ================")
fs = read("OASIS3_Freesurfer_output.csv")
log("[FreeSurfer] rows:", len(fs)); profile("FS QC Status", fs["FS QC Status"]); profile("version", fs["version"])
fs["sid"] = fs["Subject"].map(subj_from); fs["day"] = fs["MR_session"].map(days_from)
fs = fs[~fs["FS QC Status"].astype(str).str.lower().str.contains("fail|reject|unusable|quarantin")]
icv = num(fs["IntraCranialVol"])
vol_cols = [c for c in fs.columns if c.endswith("_volume") or c in ("CortexVol", "SubCortGrayVol", "TotalGrayVol",
                                                                        "CorticalWhiteMatterVol", "SupraTentorialVol")]
thk_cols = [c for c in fs.columns if c.endswith("_thickness")]
cols_ = {"sid": fs["sid"], "day": fs["day"]}
cols_.update({"V_" + c: num(fs[c]) / icv * 1000 for c in vol_cols})
cols_.update({"T_" + c: num(fs[c]) for c in thk_cols})
F = pd.DataFrame(cols_)
F["AS_HIPP"] = (num(fs["Left-Hippocampus_volume"]) + num(fs["Right-Hippocampus_volume"])) / icv * 1000
F["AS_AMYG"] = (num(fs["Left-Amygdala_volume"]) + num(fs["Right-Amygdala_volume"])) / icv * 1000
F["AS_ENT"] = (num(fs["lh_entorhinal_thickness"]) + num(fs["rh_entorhinal_thickness"])) / 2
F["AS_SIG"] = pd.concat([num(fs[f"{h}_{r}_thickness"]) for r in ("entorhinal", "inferiortemporal", "middletemporal", "fusiform")
                         for h in ("lh", "rh")], axis=1).mean(axis=1, skipna=False)
F["AS_VENT"] = sum(num(fs[c]) for c in ("Left-Lateral-Ventricle_volume", "Right-Lateral-Ventricle_volume",
                                         "Left-Inf-Lat-Vent_volume", "Right-Inf-Lat-Vent_volume")) / icv * 1000
F = F[F["day"].notna() & F["sid"].notna()]
COMPACT = ["AS_HIPP", "AS_AMYG", "AS_ENT", "AS_SIG", "AS_VENT"]
ALLFS = [c for c in F.columns if c.startswith(("V_", "T_"))]
log(f"  full FreeSurfer panel: {len(ALLFS)} measures (ICV-normalised volumes + cortical thickness)")
a2 = dA.copy(); a2["aid"] = np.arange(len(a2))
fm = nearest(a2, F, COMPACT + ALLFS, require="AS_HIPP")
a2 = a2.join(fm[COMPACT + ALLFS], on="aid")
k = a2["AS_HIPP"].notna().values
B = a2[k].reset_index(drop=True); yB = yA[k]
log(f"  amyloid cohort with FreeSurfer within {WINDOW} days of PET: {len(B)} of {len(a2)}")
if len(B) >= 50:
    Pf = amy_proba(B)
    add("Aim2 amyloid (FreeSurfer subset)", "frozen ADNI portable", "portable (frozen)", yB, Pf, CLS)
    sets = {"portable": PORT, "compact AD-signature (5)": COMPACT, "full FreeSurfer panel": ALLFS,
            "portable + compact (5)": PORT + COMPACT, "portable + full FreeSurfer": PORT + ALLFS}
    PP = {}
    for name, cols in sets.items():
        PP[name] = cv_proba(B[cols], yB, CLS)
        add("Aim2 amyloid (refitted in OASIS, 5-fold CV)", "FreeSurfer subset", name, yB, PP[name], CLS)
    deltas = []
    for aa, bb in (("portable", "portable + compact (5)"), ("portable", "portable + full FreeSurfer")):
        e, lo, hi, pv = paired_delta(yB, PP[aa], PP[bb], CLS)
        deltas.append(dict(task="amyloid", comparison=f"{bb} minus {aa}", delta=round(e, 3), ci_lo=round(lo, 3),
                           ci_hi=round(hi, 3), p=round(pv, 4)))
        log(f"  delta AUC {bb} - {aa}: {e:+.3f} ({lo:+.3f} to {hi:+.3f}), p={pv:.3f}")
    pd.DataFrame(deltas).to_csv(os.path.join(OUT, "oasis_fs_deltas_amyloid.csv"), index=False)

# ============================ Aim 3: trajectory ============================
log("\n================ Aim 3: frozen ADNI trajectory model ================")
log("ADNI reference:", FROZEN["models"]["trajectory"]["adni_cv_auc"], FROZEN["models"]["trajectory"]["adni_cv_auc_ci"])
V = vis.sort_values(["sid", "day"]).copy()
V["prior_cdr1"] = V.groupby("sid")["CDRTOT"].transform(lambda s: (s >= 1).cummax())


def trajectory_cohort(base_col, event):
    """Baseline = first visit meeting base_col (CDR 0.5) with MMSE>=24 and no earlier CDR>=1.
    fast: event <= 24 months; slow: 24-48 months; stable: no event and >= 48 months of follow-up."""
    cand = V[V[base_col] & ~V["prior_cdr1"]].drop_duplicates("sid").copy()
    cand["aid"] = np.arange(len(cand))
    mm = nearest(cand, vis, ["MMSE"], require="MMSE")
    cand = cand.drop(columns=["MMSE"]).join(mm[["MMSE"]], on="aid")
    cand = cand[cand["MMSE"] >= 24]
    ev_flag = event(V)
    labs = []
    for sid, bday in zip(cand["sid"], cand["day"]):
        k = (V["sid"] == sid) & (V["day"] > bday)
        ev = V.loc[k & ev_flag, "day"]
        fu = (V.loc[k, "day"].max() - bday) / 30.44 if k.any() else 0.0
        if len(ev):
            mo = (ev.min() - bday) / 30.44
            labs.append("fast" if mo <= 24 else ("slow" if mo <= 48 else None))
        else:
            labs.append("stable" if fu >= 48 else None)
    cand["traj"] = labs
    T = cand[cand["traj"].notna()].copy()
    T = T.join(demo[["sex_female", "PTEDUCAT", "APOE4", "AgeatEntry"]], on="sid")
    T["age"] = T["age_visit"].fillna(T["AgeatEntry"] + T["day"] / 365.25)
    log(f"  baselines {len(cand)}; labelled trajectories {len(T)}: {T['traj'].value_counts().to_dict()}")
    return T.reset_index(drop=True)


TRAJ_DEFS = {"primary: CDR 0.5 & MMSE>=24 baseline; event CDR>=1":
                 ("cdr05", lambda d: d["CDRTOT"] >= 1),
             "sensitivity: clinician 'not demented' baseline; event CDR>=1 or clinician dementia":
                 ("cdr05_notdem", lambda d: (d["CDRTOT"] >= 1) | (d["DEMENTED"] == 1))}
TT = {}
for label, (bcol, event) in TRAJ_DEFS.items():
    log(" ", label)
    T = trajectory_cohort(bcol, event); TT[label] = T
    if T["traj"].nunique() == 3 and len(T) >= 60:
        PT, clsT = frozen_proba("trajectory", T)
        add("Aim3 trajectory (frozen ADNI portable model)", label, "portable", T["traj"].values, PT, clsT)
        for kc, c in enumerate(clsT):
            a1 = bin_auc(T["traj"].values == c, PT[:, kc]); log(f"    {c} vs rest AUC {a1:.3f}")
            ROWS.append(dict(analysis="Aim3 trajectory one-vs-rest", subset=f"{label}: {c} vs rest", model="portable",
                             n=len(T), auc=round(a1, 3)))
        PTr = cv_proba(T[PORT], T["traj"].values, clsT)
        add("Aim3 trajectory (refitted in OASIS, 5-fold CV)", label, "portable (refit)", T["traj"].values, PTr, clsT)

# FreeSurfer increment for the trajectory, refitted in OASIS (primary trajectory cohort)
T = TT[list(TRAJ_DEFS)[0]].copy(); T["aid"] = np.arange(len(T))
fmT = nearest(T, F, COMPACT + ALLFS, require="AS_HIPP")
T = T.join(fmT[COMPACT + ALLFS], on="aid")
BT = T[T["AS_HIPP"].notna()].reset_index(drop=True)
log(f"  trajectory cohort with FreeSurfer within {WINDOW} days of baseline: {len(BT)} of {len(T)}")
if len(BT) >= 50 and BT["traj"].nunique() == 3:
    clsT = sorted(BT["traj"].unique()); yT = BT["traj"].values; PPT = {}
    for name, cols in {"portable": PORT, "compact AD-signature (5)": COMPACT, "full FreeSurfer panel": ALLFS,
                       "portable + compact (5)": PORT + COMPACT, "portable + full FreeSurfer": PORT + ALLFS}.items():
        PPT[name] = cv_proba(BT[cols], yT, clsT)
        add("Aim2b trajectory (refitted in OASIS, 5-fold CV)", "FreeSurfer subset", name, yT, PPT[name], clsT)
    dT = []
    for aa, bb in (("portable", "portable + compact (5)"), ("portable", "portable + full FreeSurfer")):
        e, lo, hi, pv = paired_delta(yT, PPT[aa], PPT[bb], clsT)
        dT.append(dict(task="trajectory", comparison=f"{bb} minus {aa}", delta=round(e, 3), ci_lo=round(lo, 3),
                       ci_hi=round(hi, 3), p=round(pv, 4)))
        log(f"  trajectory delta AUC {bb} - {aa}: {e:+.3f} ({lo:+.3f} to {hi:+.3f}), p={pv:.3f}")
    pd.DataFrame(dT).to_csv(os.path.join(OUT, "oasis_fs_deltas_trajectory.csv"), index=False)

# ============================ Aim 4: four groups ============================
log("\n================ Aim 4: frozen ADNI four-group model ================")
log("ADNI reference:", FROZEN["models"]["fourgroup"]["adni_cv_auc"], FROZEN["models"]["fourgroup"]["adni_cv_auc_ci"])
# first clinical visit of each participant: CN = CDR 0; AD dementia = AD-type clinical dementia with
# CDR >= 1, or CDR 0.5 with MMSE < 24 (so that it does not overlap the ADNI-harmonised MCI definition)
first = V.drop_duplicates("sid").copy(); first["aid"] = np.arange(len(first))
mmF = nearest(first, vis, ["MMSE"], require="MMSE")
first = first.drop(columns=["MMSE"]).join(mmF[["MMSE"]], on="aid")
ad_flag = pd.Series(False, index=first.index)
for c in ("PROBAD", "POSSAD", "alzdis"):
    if c in first:
        ad_flag |= first[c] == 1
ad_txt = first["dx1"].map(lambda s: bool(AD_TXT.search(s)) and bool(DEM_TXT.search(s)) and not s.lower().startswith("unc"))
ad_dx = first["demented"].astype(bool) & (ad_flag | ad_txt)
severity = (first["CDRTOT"] >= 1) | ((first["CDRTOT"] == 0.5) & (first["MMSE"] < 24))
first["grp"] = np.where((first["CDRTOT"] == 0) & ~first["demented"].astype(bool), "CN",
                        np.where(ad_dx & severity, "AD", None))
mci_ids = set(dA["sid"])
G = first[first["grp"].notna() & ~first["sid"].isin(mci_ids)].copy()
G = G.join(demo[["sex_female", "PTEDUCAT", "APOE4", "AgeatEntry"]], on="sid")
G["age"] = G["age_visit"].fillna(G["AgeatEntry"] + G["day"] / 365.25)
M4 = dA.copy(); M4["grp"] = np.where(yA == "A+", "A+MCI", "A-MCI")
G4 = pd.concat([G[PORT + ["grp"]], M4[PORT + ["grp"]]], ignore_index=True)
log(f"  four-group counts: {G4['grp'].value_counts().to_dict()}")
if G4["grp"].nunique() == 4:
    P4, cls4 = frozen_proba("fourgroup", G4)
    add("Aim4 four-group (frozen ADNI portable model)", "CN / A-MCI / A+MCI / AD", "portable", G4["grp"].values, P4, cls4)
    for kc, c in enumerate(cls4):
        a1 = bin_auc(G4["grp"].values == c, P4[:, kc]); log(f"    {c} vs rest AUC {a1:.3f}")
        ROWS.append(dict(analysis="Aim4 four-group one-vs-rest", subset=f"{c} vs rest", model="portable",
                         n=len(G4), auc=round(a1, 3)))

# ============================ save ============================
res = pd.DataFrame(ROWS)
res.to_csv(os.path.join(OUT, "oasis_results.csv"), index=False)
log("\nSaved aggregate results to", OUT)
log("Files: oasis_results.csv, oasis_table1.csv, oasis_calibration_summary.csv, oasis_calibration_deciles.csv,")
log("       oasis_decision_curve.csv, oasis_operating_points.csv, oasis_fs_deltas_amyloid.csv, oasis_fs_deltas_trajectory.csv, oasis_log.txt")
LOGF.close()
