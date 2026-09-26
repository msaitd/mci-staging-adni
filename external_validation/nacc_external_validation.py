"""NACC external validation of the frozen ADNI 'portable' models (Paper B, revision).

Run locally by the authorised NACC data user:
    python nacc_external_validation.py

Inputs : NACC Quick-Access files (June 2026 freeze) under DATA_ROOT
         frozen_adni_models.json               (ADNI model parameters only; same folder as this script)
         moca_mmse_crosswalk.csv               (education-adjusted MoCA -> equivalent MMSE with 95% CI; NACC Crosswalk
                                                Study, Monsell et al. 2016, Alzheimer Dis Assoc Disord 30:134-139, Table 3)
Outputs: ./nacc_results/ -- AGGREGATE results only (no participant IDs, no participant-level values;
         counts < 5 are masked in the log)

Pre-specified analyses (mirroring the OASIS-3 validation)
  Aim 1  frozen amyloid model in MCI with SCAN/CLARiTI amyloid PET (Centiloid >= 20); sensitivity analyses
  Aim 2  incremental value of FreeSurfer (SCAN MRI) measures, refitted within NACC (5-fold CV, paired bootstrap)
  Aim 3  frozen trajectory model (stable / slow / fast progression to clinician-diagnosed dementia)
  Aim 4  frozen four-group model (CN / A-MCI / A+MCI / AD dementia)
  Aim 5  autopsy: frozen amyloid model against neuritic plaques (CERAD) and NIA-AA ADNC in MCI
"""
import os, re, sys, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# folder with the NACC Quick-Access files (UDS, SCAN PET/MRI); set NACC_ROOT or edit this line
DATA_ROOT = os.environ.get("NACC_ROOT", os.path.join(os.path.dirname(HERE), "nacc_data"))
OUT = os.path.join(HERE, "nacc_results"); os.makedirs(OUT, exist_ok=True)
FROZEN = json.load(open(os.path.join(HERE, "frozen_adni_models.json"), encoding="utf-8"))
PORT = FROZEN["features"]
WINDOW = 365
N_BOOT = 2000
SEED = 42
LOGF = open(os.path.join(OUT, "nacc_log.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOGF.write(s + "\n"); LOGF.flush()


def find(pattern):
    hits = [h for h in glob.glob(os.path.join(DATA_ROOT, "**", pattern), recursive=True)]
    if not hits:
        sys.exit(f"File not found under {DATA_ROOT}: {pattern}")
    return sorted(hits)[0]


num = lambda s: pd.to_numeric(s, errors="coerce")
mask = lambda vc: {k: (int(v) if v >= 5 else "<5") for k, v in dict(vc).items()}   # small-cell suppression
cell = lambda v: int(v) if v >= 5 or v == 0 else "<5"


def profile(name, s, top=25):
    if s.dtype == object or s.nunique(dropna=True) <= 15:
        vc = s.astype(str).value_counts(dropna=False).head(top)
        log(f"  [{name}] " + "; ".join(f"{k}={v if v >= 5 else '<5'}" for k, v in vc.items()))
    else:
        v = num(s)
        log(f"  [{name}] n={v.notna().sum()} missing={v.isna().mean():.2%} min={v.min():.3g} median={v.median():.3g} max={v.max():.3g}")


def valid(s, lo, hi):
    v = num(s); return v.where((v >= lo) & (v <= hi))


# ============================ statistics (identical to the OASIS-3 script) ============================
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


def boot(y, P, classes, n=N_BOOT, seed=1):
    y = np.asarray(y).astype(str); rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < len(classes):
            continue
        vals.append(auc_any(y[s], P[s], classes))
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def paired_delta(y, PA, PB, classes, n=N_BOOT, seed=2):
    y = np.asarray(y).astype(str); rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < len(classes):
            continue
        d.append(auc_any(y[s], PB[s], classes) - auc_any(y[s], PA[s], classes))
    d = np.array(d); est = auc_any(y, PB, classes) - auc_any(y, PA, classes)
    return est, np.percentile(d, 2.5), np.percentile(d, 97.5), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def logistic_fit(X, y, offset=None, iters=100):
    X = np.column_stack([np.ones(len(y))] + ([X] if X is not None else []))
    off = np.zeros(len(y)) if offset is None else offset
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ b + off))); W = mu * (1 - mu) + 1e-12
        step = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (y - mu)); b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def calibration(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); lp = logit(p)
    out = dict(brier=float(np.mean((p - y) ** 2)), cal_slope=np.nan, cal_intercept_citl=np.nan,
               mean_pred=float(p.mean()), observed=float(y.mean()))
    if len(np.unique(y)) < 2:
        return out
    try:
        a, b = logistic_fit(lp, y); citl = logistic_fit(None, y, offset=lp)[0]
        out.update(cal_slope=float(b), cal_intercept_citl=float(citl))
    except np.linalg.LinAlgError:
        pass
    return out


def net_benefit(y, p, t):
    y = np.asarray(y, bool); pos = p >= t
    return (np.sum(pos & y) - np.sum(pos & ~y) * t / (1 - t)) / len(y)


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
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    Xv = Xdf.apply(num).values.astype(float); y = np.asarray(y).astype(str)
    Xv = Xv[:, ~np.all(np.isnan(Xv), axis=0)]
    P = np.full((len(y), len(classes)), np.nan)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Xv, y):
        m = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]).fit(Xv[tr], y[tr])
        cl = list(m.classes_); P[te] = m.predict_proba(Xv[te])[:, [cl.index(c) for c in classes]]
    return P


ROWS = []
CLS = ["A-", "A+"]


MIN_N, MIN_CLASS = 30, 5


def enough(y, classes):
    y = np.asarray(y).astype(str)
    return len(y) >= MIN_N and all((y == c).sum() >= MIN_CLASS for c in classes)


def add(analysis, subset, model, y, P, classes, extra=None):
    y = np.asarray(y).astype(str)
    if not enough(y, classes):
        log(f"  {analysis} | {subset} | {model}: SKIPPED (n={len(y)}; needs n>={MIN_N} and >={MIN_CLASS} per class)")
        return None
    a = auc_any(y, P, classes); lo, hi = boot(y, P, classes)
    r = dict(analysis=analysis, subset=subset, model=model, n=len(y),
             class_counts="; ".join(f"{c}={int((y == c).sum())}" for c in classes),
             auc=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3))
    if len(classes) == 2:
        r.update({k: round(v, 3) for k, v in calibration((y == classes[1]).astype(int), P[:, 1]).items()})
    if extra:
        r.update(extra)
    ROWS.append(r); log(f"  {analysis} | {subset} | {model}: n={len(y)} AUC {a:.3f} ({lo:.3f}-{hi:.3f})")
    return r


def amy_proba(df):
    P, cls = frozen_proba("amyloid", df); pA = P[:, cls.index("A+")]
    return np.column_stack([1 - pA, pA])


def nearest(anchor, table, cols, require=None, window=WINDOW):
    """For each anchor row (aid, sid, date) the table row of the same participant nearest in time (|days| <= window)."""
    t = table[["sid", "date"] + cols].copy()
    if require:
        t = t[t[require].notna()]
    m = anchor[["aid", "sid", "date"]].merge(t, on="sid", suffixes=("", "_t"))
    m["gap"] = (m["date_t"] - m["date"]).dt.days.abs()
    m = m[m["gap"] <= window].sort_values(["aid", "gap"]).drop_duplicates("aid")
    return m.set_index("aid")


# ============================ load UDS ============================
log("NACC external validation -- aggregate log (no participant-level data)")
log("DATA_ROOT:", DATA_ROOT)
UCOLS = ["NACCID", "NACCADC", "PACKET", "FORMVER", "VISITDATE", "NACCVNUM", "NACCFDYS", "NACCAGE", "NACCSEX", "EDUC",
         "NACCNE4S", "NACCMMSE", "MOCATOTS", "NACCMOCA", "CDRGLOB", "CDRSUM", "NACCUDSD", "DEMENTED", "NACCALZP",
         "NACCETPR", "NACCDIED", "NACCMOD", "NACCYOD", "NACCINT", "NACCNEUR", "NPADNC", "NPTHAL"]
ufile = find("investigator_nacc*.csv")
uds = pd.read_csv(ufile, usecols=lambda c: c in UCOLS, low_memory=False)
log(f"\n[UDS] {os.path.basename(ufile)}: {len(uds)} visits, {uds['NACCID'].nunique()} participants")
for c in ("PACKET", "FORMVER", "NACCSEX", "NACCNE4S", "CDRGLOB", "NACCUDSD", "DEMENTED", "NACCALZP", "NACCNEUR", "NPADNC"):
    profile(c, uds[c])
for c in ("NACCAGE", "EDUC", "NACCMMSE", "MOCATOTS", "NACCMOCA", "CDRSUM"):
    profile(c, uds[c])
log("  VISITDATE examples of format:", uds["VISITDATE"].dropna().astype(str).str.len().value_counts().head(3).to_dict())
uds["sid"] = uds["NACCID"].astype(str)
uds["date"] = pd.to_datetime(uds["VISITDATE"], errors="coerce")
log(f"  VISITDATE parsed: {uds['date'].notna().mean():.1%}")
uds["age"] = valid(uds["NACCAGE"], 18, 120)
uds["sex_female"] = num(uds["NACCSEX"]).map({1: 0.0, 2: 1.0})
uds["PTEDUCAT"] = valid(uds["EDUC"], 0, 36)
uds["APOE4"] = valid(uds["NACCNE4S"], 0, 2)
uds["CDRSB"] = valid(uds["CDRSUM"], 0, 18)
uds["CDR"] = num(uds["CDRGLOB"]).where(num(uds["CDRGLOB"]).isin([0, 0.5, 1, 2, 3]))
uds["mmse_obs"] = valid(uds["NACCMMSE"], 0, 30)
moca_adj = valid(uds["NACCMOCA"], 0, 30)
moca_raw = valid(uds["MOCATOTS"], 0, 30)
uds["moca"] = moca_adj.fillna((moca_raw + (uds["PTEDUCAT"] <= 12).astype(float)).clip(upper=30))
# MoCA -> MMSE equipercentile crosswalk (NACC Crosswalk Study), if the table is supplied
cw_path = os.path.join(HERE, "moca_mmse_crosswalk.csv")
if os.path.exists(cw_path):
    cw = pd.read_csv(cw_path); CW = dict(zip(cw["moca"].astype(int), cw["mmse"].astype(float)))
    uds["mmse_from_moca"] = uds["moca"].round().map(CW)
    log("  MoCA->MMSE crosswalk loaded:", len(CW), "entries")
else:
    uds["mmse_from_moca"] = np.nan
    log("  WARNING: moca_mmse_crosswalk.csv not found -- MMSE is available only where it was administered")
uds["MMSE"] = uds["mmse_obs"].fillna(uds["mmse_from_moca"])
uds["mmse_source"] = np.where(uds["mmse_obs"].notna(), "MMSE", np.where(uds["mmse_from_moca"].notna(), "MoCA->MMSE", "none"))
if uds["mmse_from_moca"].notna().any():
    # Plausibility check of the crosswalk: last observed MMSE (UDS2) versus the converted MoCA of the next visit (UDS3)
    # of the same participant within 365 days. Aggregate agreement only.
    s = uds[["NACCID", "VISITDATE", "mmse_obs", "mmse_from_moca"]].copy()
    s["date"] = pd.to_datetime(s["VISITDATE"], errors="coerce"); s = s.sort_values(["NACCID", "date"])
    s["next_conv"] = s.groupby("NACCID")["mmse_from_moca"].shift(-1); s["next_date"] = s.groupby("NACCID")["date"].shift(-1)
    s["next_obs"] = s.groupby("NACCID")["mmse_obs"].shift(-1)
    pr = s[s["mmse_obs"].notna() & s["next_obs"].isna() & s["next_conv"].notna() & ((s["next_date"] - s["date"]).dt.days <= 365)]
    pr = pr.drop_duplicates("NACCID", keep="last")
    if len(pr) >= 30:
        dlt = pr["next_conv"] - pr["mmse_obs"]
        log(f"  crosswalk check (MMSE visit -> MoCA visit <=365 d, n={len(pr)}): converted minus observed "
            f"mean {dlt.mean():+.2f} (SD {dlt.std():.2f}); exact {np.mean(dlt == 0):.0%}, within 1 {np.mean(dlt.abs() <= 1):.0%}, "
            f"within 2 {np.mean(dlt.abs() <= 2):.0%}; Spearman rho {pr['mmse_obs'].corr(pr['next_conv'], method='spearman'):.2f}")
        pd.DataFrame([dict(n=len(pr), mean_diff=dlt.mean(), sd_diff=dlt.std(), exact=np.mean(dlt == 0), within1=np.mean(dlt.abs() <= 1),
                           within2=np.mean(dlt.abs() <= 2), spearman=pr["mmse_obs"].corr(pr["next_conv"], method="spearman"))]).round(3).to_csv(
            os.path.join(OUT, "nacc_crosswalk_check.csv"), index=False)
uds["demented"] = (num(uds["NACCUDSD"]) == 4)
uds["mci_clin"] = (num(uds["NACCUDSD"]) == 3)
uds["ad_primary"] = (num(uds["NACCALZP"]) == 1) | (num(uds["NACCETPR"]) == 1)
uds = uds[uds["date"].notna()].sort_values(["sid", "date"]).reset_index(drop=True)
log(f"  MMSE source across visits: {uds['mmse_source'].value_counts().to_dict()}")

# MCI definitions (visit level; MMSE criterion applied after taking the nearest MMSE within the window)
uds["cdr05"] = uds["CDR"] == 0.5
uds["cdr05_notdem"] = uds["cdr05"] & ~uds["demented"]
MCI_DEFS = {"primary: CDR 0.5, MMSE>=24, not clinically demented (ADNI-harmonised)": ("cdr05_notdem", 24),
            "sensitivity: clinician MCI diagnosis (NACCUDSD=3)": ("mci_clin", None),
            "sensitivity: CDR 0.5, MMSE>=24 (clinician label ignored)": ("cdr05", 24),
            "sensitivity: CDR 0.5, not demented, any MMSE": ("cdr05_notdem", None)}
PRIMARY = list(MCI_DEFS)[0]
demo_cols = ["sid", "date", "age", "sex_female", "PTEDUCAT", "APOE4", "CDRSB", "CDR", "MMSE", "mmse_source"]

# ============================ amyloid PET ============================
# TRACER codes of the SCAN/CLARiTI MRI-free amyloid PET GAAIN file (NACC SCAN & CLARiTI PET Researcher's Data Dictionary v4.0)
TRACER_RDD = {1: "FDG", 2: "PiB", 3: "florbetapir", 4: "florbetaben", 5: "NAV4694", 10: "flutemetamol", 99: "unknown"}


def load_pet(pattern, source, tracer_map=None):
    f = glob.glob(os.path.join(DATA_ROOT, "**", pattern), recursive=True)
    if not f:
        log("  missing:", pattern); return pd.DataFrame()
    d = pd.read_csv(f[0], low_memory=False); d.columns = [c.upper() for c in d.columns]
    log(f"\n[{source}] {os.path.basename(f[0])}: {len(d)} scans")
    for c in [c for c in ("TRACER", "QCSTATUS", "QC_IMAGE", "QC_TIMING", "CL_FAIL", "AMYLOIDSTATUS", "AMYLOID_STATUS", "PROJECT", "SCAN_PROJECT") if c in d]:
        profile(c, d[c])
    trn = num(d["TRACER"]).astype("Int64")
    if tracer_map:                                    # names from the data dictionary
        tracer = trn.map(tracer_map).fillna("code " + trn.astype(str)).astype(str)
    else:                                             # codes kept as codes where the dictionary was not checked
        tracer = ("code " + trn.astype(str)) if trn.notna().mean() > 0.9 else d["TRACER"].astype(str).str.upper()
    out = pd.DataFrame({"sid": d["NACCID"].astype(str), "date": pd.to_datetime(d["SCANDATE"], errors="coerce"),
                        "CL": num(d["CENTILOIDS"]), "tracer": tracer, "source": source,
                        "loniuid": d["LONIUID"].astype(str).str.replace(r"\.0$", "", regex=True) if "LONIUID" in d else ""})
    st = d["AMYLOIDSTATUS"] if "AMYLOIDSTATUS" in d else d.get("AMYLOID_STATUS")
    # SCAN amyloid status: 1 = elevated (A+), 0 = not elevated (A-), anything else (e.g. 9) = not available
    out["scan_status"] = num(st).map({1: "A+", 0: "A-"}) if st is not None else np.nan
    if "QCSTATUS" in d:                               # standardized SCAN/CLARiTI pipeline: 1 = QC passed
        out["qc_ok"] = num(d["QCSTATUS"]) == 1
        out["qc_strict"] = out["qc_ok"]
    else:                                             # mixed protocol: QC_IMAGE / QC_TIMING flags, 1 = passed
        qi = num(d["QC_IMAGE"]) == 1 if "QC_IMAGE" in d else pd.Series(True, index=d.index)
        qt = num(d["QC_TIMING"]) == 1 if "QC_TIMING" in d else pd.Series(True, index=d.index)
        out["qc_ok"] = qi                             # image QC required; acquisition-timing QC in a sensitivity analysis
        out["qc_strict"] = qi & qt                    # CL_FAIL holds numeric values (not a 0/1 flag) and is not used
    profile("Centiloids", out["CL"]); log(f"  SCANDATE parsed: {out['date'].notna().mean():.1%}")
    return out[out["CL"].notna() & out["date"].notna()]


pet_std = load_pet("*scan_clariti_amyloidpetgaain*.csv", "SCAN/CLARiTI standardized", TRACER_RDD)
pet_mp = load_pet("*scan_mp_amyloidpetgaain*.csv", "mixed protocol")
PET = pd.concat([pet_std, pet_mp], ignore_index=True)
log(f"\nAmyloid PET with Centiloid: standardized {len(pet_std)} scans ({pet_std['sid'].nunique()} participants); "
    f"mixed protocol {len(pet_mp)} ({pet_mp['sid'].nunique() if len(pet_mp) else 0})")
# tracer code -> radiotracer name, read from the SCAN PET QC file (linked on LONIUID); aggregate cross-tabulation only
qcf = glob.glob(os.path.join(DATA_ROOT, "**", "*scan_clariti_petqc*.csv"), recursive=True)
if qcf:
    q = pd.read_csv(qcf[0], low_memory=False, usecols=lambda c: c.upper() in ("LONIUID", "RADIOTRACER"))
    q.columns = [c.upper() for c in q.columns]; q["LONIUID"] = q["LONIUID"].astype(str).str.replace(r"\.0$", "", regex=True)
    xt = PET.merge(q.drop_duplicates("LONIUID"), left_on="loniuid", right_on="LONIUID", how="left")
    tab = xt.groupby(["tracer", xt["RADIOTRACER"].astype(str)]).size()
    log("  tracer code x RADIOTRACER (SCAN PET QC file): " + "; ".join(f"{a} / {b}: {n if n >= 5 else '<5'}" for (a, b), n in tab.items()))


def amyloid_cohort(defn, pet):
    col, mmin = defn
    a = pet.reset_index(drop=True).copy(); a["aid"] = np.arange(len(a))
    v = nearest(a, uds, ["age", "CDRSB", "CDR", col])
    a = a.join(v[["age", "CDRSB", "CDR", col, "gap"]], on="aid")
    mm = nearest(a, uds, ["MMSE", "mmse_source"], require="MMSE")
    a = a.join(mm[["MMSE", "mmse_source"]], on="aid")
    ok = a[col] == True
    if mmin is not None:
        ok &= a["MMSE"] >= mmin
    a = a[ok].sort_values(["sid", "date"]).drop_duplicates("sid")
    stat = uds.drop_duplicates("sid", keep="last").set_index("sid")[["sex_female", "PTEDUCAT", "APOE4"]]
    first = uds.groupby("sid")[["sex_female", "PTEDUCAT", "APOE4"]].first()
    a = a.join(first.combine_first(stat), on="sid")
    return a.reset_index(drop=True)


def amyloid_eval(Adf, thr, subset, cl_col="CL"):
    d = Adf[Adf[cl_col].notna()]
    y = np.where(d[cl_col] >= thr, "A+", "A-"); P = amy_proba(d)
    return add("Aim1 amyloid (frozen ADNI portable model)", subset, "portable", y, P, CLS), d, y, P


log("\n================ Aim 1: frozen ADNI amyloid model ================")
log("ADNI reference:", FROZEN["models"]["amyloid"]["adni_cv_auc"], FROZEN["models"]["amyloid"]["adni_cv_auc_ci"])
std_qc = pet_std[pet_std["qc_ok"]]
cohorts = {}
for label, defn in MCI_DEFS.items():
    A = amyloid_cohort(defn, std_qc); cohorts[label] = A
    log(f"  {label}: {len(A)} participants; PET-to-visit gap median {A['gap'].median():.0f} days; "
        f"tracers {mask(A['tracer'].value_counts())}; MMSE source {mask(A['mmse_source'].value_counts())}")
A = cohorts[PRIMARY]
log("Predictor missingness in the primary cohort (imputed with ADNI training medians):")
for c in PORT:
    log(f"  {c}: {A[c].isna().mean():.1%} missing")

_, dA, yA, PA = amyloid_eval(A, 20, "primary cohort, Centiloid >= 20")
for thr in (12, 24.1, 30):
    amyloid_eval(A, thr, f"primary cohort, Centiloid >= {thr}")
k = A["scan_status"].isin(["A+", "A-"])
if k.sum() >= MIN_N:
    d = A[k]
    add("Aim1 amyloid (frozen ADNI portable model)", "SCAN-provided amyloid status (AMYLOIDSTATUS)", "portable", d["scan_status"].values, amy_proba(d), CLS)
    agree = np.mean((d["CL"] >= 20) == (d["scan_status"] == "A+"))
    log(f"  agreement of Centiloid >= 20 with SCAN-provided status: {agree:.1%} (n={len(d)})")
for tr in sorted(A["tracer"].unique()):
    if (A["tracer"] == tr).sum() >= 30:
        amyloid_eval(A[A["tracer"] == tr], 20, f"tracer {tr}")
amyloid_eval(A[A["mmse_source"] == "MMSE"], 20, "MMSE administered (no MoCA conversion)") if (A["mmse_source"] == "MMSE").sum() >= 30 else None
amyloid_eval(A[A["gap"] <= 90], 20, "clinical visit within 90 days of PET")
if (A["mmse_source"] == "MoCA->MMSE").any():   # effect of the crosswalk: MoCA-derived MMSE removed (imputed with the ADNI median)
    A_noc = A.copy(); A_noc.loc[A_noc["mmse_source"] == "MoCA->MMSE", "MMSE"] = np.nan
    amyloid_eval(A_noc, 20, "MoCA-derived MMSE not used (missing MMSE imputed with the ADNI median)")
    # crosswalk uncertainty: MoCA-derived MMSE replaced by the lower / upper 95% CI bound of the published table
    # (affects both the MMSE >= 24 entry criterion and the MMSE predictor)
    for bound, lab_ in (("ci_lo", "lower"), ("ci_hi", "upper")):
        if bound in cw:
            keep = uds["MMSE"].copy()
            uds["MMSE"] = uds["mmse_obs"].fillna(uds["moca"].round().map(dict(zip(cw["moca"].astype(int), cw[bound].astype(float)))))
            Ab = amyloid_cohort(MCI_DEFS[PRIMARY], std_qc); uds["MMSE"] = keep
            amyloid_eval(Ab, 20, f"MoCA-derived MMSE at the {lab_} 95% CI bound of the crosswalk")
for label, Ad in cohorts.items():
    if label != PRIMARY and len(Ad) >= 30:
        amyloid_eval(Ad, 20, label)
if len(pet_mp):
    Amp = amyloid_cohort(MCI_DEFS[PRIMARY], pd.concat([std_qc, pet_mp[pet_mp["qc_ok"]]], ignore_index=True))
    amyloid_eval(Amp, 20, "standardized + mixed-protocol PET (image QC passed)")
    Amp2 = amyloid_cohort(MCI_DEFS[PRIMARY], pd.concat([std_qc, pet_mp[pet_mp["qc_strict"]]], ignore_index=True))
    amyloid_eval(Amp2, 20, "standardized + mixed-protocol PET (image and timing QC passed)")
    Amp3 = amyloid_cohort(MCI_DEFS[PRIMARY], pet_mp[pet_mp["qc_ok"]])
    amyloid_eval(Amp3, 20, "mixed-protocol PET only (image QC passed)")
for lab_, k in (("APOE e4 non-carriers", A["APOE4"] == 0), ("APOE e4 carriers", A["APOE4"] >= 1)):
    if k.sum() >= 30:
        amyloid_eval(A[k], 20, lab_)
if len(A) >= 90:
    tert = pd.qcut(A["age"], 3, labels=False, duplicates="drop")
    for t in sorted(tert.dropna().unique()):
        k = tert == t
        amyloid_eval(A[k], 20, f"age tertile {int(t) + 1} ({A.loc[k, 'age'].min():.0f}-{A.loc[k, 'age'].max():.0f} y)")
PRIMARY_OK = enough(yA, CLS)
if not PRIMARY_OK:
    log(f"\n  PRIMARY AMYLOID COHORT TOO SMALL (n={len(yA)}): calibration, operating points, Table 1 and the FreeSurfer analysis "
        "are skipped. In the MoCA era (UDS v3 onwards) MMSE is available only through moca_mmse_crosswalk.csv.")
else:
    Pref = cv_proba(dA[PORT], yA, CLS)
    add("Aim1 amyloid (refitted in NACC, 5-fold CV)", "primary cohort, Centiloid >= 20", "portable (refit)", yA, Pref, CLS)

if PRIMARY_OK:
    p = PA[:, 1]; yb = (yA == "A+").astype(int)
    cal = calibration(yb, p); p_rec = 1 / (1 + np.exp(-(logit(p) + cal["cal_intercept_citl"]))); cal_r = calibration(yb, p_rec)
    ops = []
    for name, t in (("ADNI 90%-sensitivity threshold (0.25)", 0.25), ("NACC 90%-sensitivity threshold", float(np.quantile(p[yb == 1], 0.10)))):
        pos = p >= t
        tp, fp, fn, tn = (pos & (yb == 1)).sum(), (pos & (yb == 0)).sum(), (~pos & (yb == 1)).sum(), (~pos & (yb == 0)).sum()
        ops.append(dict(operating_point=name, threshold=round(t, 3), sensitivity=round(tp / (tp + fn), 3), specificity=round(tn / (tn + fp), 3),
                        ppv=round(tp / max(tp + fp, 1), 3), npv=round(tn / max(tn + fn, 1), 3), fraction_referred=round(pos.mean(), 3)))
    pd.DataFrame(ops).to_csv(os.path.join(OUT, "nacc_operating_points.csv"), index=False)
    pd.DataFrame([dict(model="frozen", **cal), dict(model="intercept-recalibrated", **cal_r)]).round(3).to_csv(
        os.path.join(OUT, "nacc_calibration_summary.csv"), index=False)
    dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    pd.DataFrame({"decile": dec, "p": p, "y": yb}).groupby("decile").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).round(3).to_csv(
        os.path.join(OUT, "nacc_calibration_deciles.csv"))
    pd.DataFrame([dict(threshold=t, model=round(net_benefit(yb, p, t), 4), model_recalibrated=round(net_benefit(yb, p_rec, t), 4),
                       test_all=round(yb.mean() - (1 - yb.mean()) * t / (1 - t), 4), test_none=0.0) for t in (0.1, 0.2, 0.3, 0.4, 0.5)]).to_csv(
        os.path.join(OUT, "nacc_decision_curve.csv"), index=False)
    log(f"  calibration (frozen): {cal} | after intercept recalibration: Brier {cal_r['brier']:.3f}")

    try:
        from scipy import stats
    except ImportError:
        stats = None
    t1 = []
    for c, lab, kind in (("age", "Age, years", "c"), ("sex_female", "Female", "b"), ("PTEDUCAT", "Education, years", "c"),
                         ("APOE4", "APOE e4 carrier", "b"), ("MMSE", "MMSE (observed or converted from MoCA)", "c"), ("CDRSB", "CDR-SB", "c"), ("CL", "Centiloid", "c")):
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
        t1.append(dict(characteristic=f"Tracer {tr}", A_pos=cell(((dA["tracer"] == tr) & (yA == "A+")).sum()), A_neg=cell(((dA["tracer"] == tr) & (yA == "A-")).sum()), p=np.nan))
    for s_ in sorted(dA["mmse_source"].astype(str).unique()):
        t1.append(dict(characteristic=f"MMSE source: {s_}", A_pos=cell(((dA["mmse_source"] == s_) & (yA == "A+")).sum()), A_neg=cell(((dA["mmse_source"] == s_) & (yA == "A-")).sum()), p=np.nan))
    pd.DataFrame(t1).to_csv(os.path.join(OUT, "nacc_table1.csv"), index=False)

# ============================ Aim 2: FreeSurfer (SCAN MRI), refitted in NACC ============================
log("\n================ Aim 2: incremental value of FreeSurfer (SCAN MRI, refitted in NACC) ================")
F, COMPACT, ALLFS = pd.DataFrame(), [], []
fsf = glob.glob(os.path.join(DATA_ROOT, "**", "*scan_clariti_mrisbm*.csv"), recursive=True)
if fsf:
    fs = pd.read_csv(fsf[0], low_memory=False); fs.columns = [c.upper() for c in fs.columns]
    log(f"[SCAN MRI FreeSurfer] {len(fs)} scans"); profile("FREESURFERVERSION", fs["FREESURFERVERSION"])
    icv = num(fs["ESTIMATEDTOTALINTRACRANIALVOL"])
    vol = [c for c in fs.columns if c.endswith("GVOL")] + [c for c in ("LEFTLATERALVENTRICLE", "RIGHTLATERALVENTRICLE", "LEFTINFLATVENT", "RIGHTINFLATVENT",
           "LEFTTHALAMUSPROPER", "RIGHTTHALAMUSPROPER", "LEFTCAUDATE", "RIGHTCAUDATE", "LEFTPUTAMEN", "RIGHTPUTAMEN", "LEFTPALLIDUM", "RIGHTPALLIDUM",
           "LEFTHIPPOCAMPUS", "RIGHTHIPPOCAMPUS", "LEFTAMYGDALA", "RIGHTAMYGDALA", "LEFTACCUMBENSAREA", "RIGHTACCUMBENSAREA", "LEFTVENTRALDC", "RIGHTVENTRALDC",
           "3RDVENTRICLE", "4THVENTRICLE", "BRAINSTEM", "CORTEX", "CEREBRALWHITEMATTER", "SUBCORTGRAY", "TOTALGRAY", "WMHYPOINTENSITIES") if c in fs.columns]
    area = [c for c in fs.columns if c.endswith("SAREA")]; thk = [c for c in fs.columns if c.endswith("AVGTH")]
    cols_ = {"sid": fs["NACCID"].astype(str), "date": pd.to_datetime(fs["SCANDT"], errors="coerce")}
    cols_.update({"V_" + c: num(fs[c]) / icv * 1000 for c in vol}); cols_.update({"A_" + c: num(fs[c]) for c in area}); cols_.update({"T_" + c: num(fs[c]) for c in thk})
    F = pd.DataFrame(cols_)
    F["AS_HIPP"] = (num(fs["LEFTHIPPOCAMPUS"]) + num(fs["RIGHTHIPPOCAMPUS"])) / icv * 1000
    F["AS_AMYG"] = (num(fs["LEFTAMYGDALA"]) + num(fs["RIGHTAMYGDALA"])) / icv * 1000
    F["AS_ENT"] = (num(fs["LHENTORHINALAVGTH"]) + num(fs["RHENTORHINALAVGTH"])) / 2
    F["AS_SIG"] = pd.concat([num(fs[f"{h}{r}AVGTH"]) for r in ("ENTORHINAL", "INFERIORTEMPORAL", "MIDDLETEMPORAL", "FUSIFORM") for h in ("LH", "RH")], axis=1).mean(axis=1, skipna=False)
    F["AS_VENT"] = sum(num(fs[c]) for c in ("LEFTLATERALVENTRICLE", "RIGHTLATERALVENTRICLE", "LEFTINFLATVENT", "RIGHTINFLATVENT")) / icv * 1000
    F = F[F["date"].notna()]
    COMPACT = ["AS_HIPP", "AS_AMYG", "AS_ENT", "AS_SIG", "AS_VENT"]; ALLFS = [c for c in F.columns if c.startswith(("V_", "A_", "T_"))]
    log(f"  full FreeSurfer panel: {len(ALLFS)} measures")


def fs_increment(D, y, classes, tag, fname):
    D = D.copy(); D["aid"] = np.arange(len(D))
    fm = nearest(D, F, COMPACT + ALLFS, require="AS_HIPP")
    D = D.join(fm[COMPACT + ALLFS], on="aid"); k = D["AS_HIPP"].notna().values
    B = D[k].reset_index(drop=True); yB = np.asarray(y)[k]
    log(f"  {tag}: FreeSurfer within {WINDOW} days available for {len(B)} of {len(D)}")
    if len(B) < 50 or not enough(yB, classes):
        log(f"  {tag}: FreeSurfer analysis skipped (too few participants or events)")
        return
    PP = {}
    for name, cols in {"portable": PORT, "compact AD-signature (5)": COMPACT, "full FreeSurfer panel": ALLFS,
                       "portable + compact (5)": PORT + COMPACT, "portable + full FreeSurfer": PORT + ALLFS}.items():
        PP[name] = cv_proba(B[cols], yB, classes)
        add(f"Aim2 {tag} (refitted in NACC, 5-fold CV)", "FreeSurfer subset", name, yB, PP[name], classes)
    rows = []
    for aa, bb in (("portable", "portable + compact (5)"), ("portable", "portable + full FreeSurfer")):
        e, lo, hi, pv = paired_delta(yB, PP[aa], PP[bb], classes)
        rows.append(dict(task=tag, comparison=f"{bb} minus {aa}", delta=round(e, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3), p=round(pv, 4)))
        log(f"  {tag} delta AUC {bb} - {aa}: {e:+.3f} ({lo:+.3f} to {hi:+.3f}), p={pv:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, fname), index=False)


if len(F) and PRIMARY_OK:
    fs_increment(dA, yA, CLS, "amyloid", "nacc_fs_deltas_amyloid.csv")

# ============================ Aim 3: trajectory ============================
log("\n================ Aim 3: frozen ADNI trajectory model ================")
log("ADNI reference:", FROZEN["models"]["trajectory"]["adni_cv_auc"], FROZEN["models"]["trajectory"]["adni_cv_auc_ci"])
V = uds.copy()
V["prior_event"] = V.groupby("sid")["demented"].transform(lambda s: s.cummax())


def trajectory_cohort(base_col, event_fn):
    cand = V[V[base_col] & ~V["prior_event"]].drop_duplicates("sid").copy(); cand["aid"] = np.arange(len(cand))
    mm = nearest(cand, uds, ["MMSE"], require="MMSE")
    cand = cand.drop(columns=["MMSE"]).join(mm[["MMSE"]], on="aid"); cand = cand[cand["MMSE"] >= 24]
    ev_flag = event_fn(V); labs = []
    g = V.groupby("sid")
    for sid, bdate in zip(cand["sid"], cand["date"]):
        f = g.get_group(sid); f = f[f["date"] > bdate]
        ev = f.loc[ev_flag.loc[f.index], "date"]
        fu = (f["date"].max() - bdate).days / 30.44 if len(f) else 0.0
        if len(ev):
            mo = (ev.min() - bdate).days / 30.44; labs.append("fast" if mo <= 24 else ("slow" if mo <= 48 else None))
        else:
            labs.append("stable" if fu >= 48 else None)
    cand["traj"] = labs
    T = cand[cand["traj"].notna()].reset_index(drop=True)
    log(f"  baselines {len(cand)}; labelled trajectories {len(T)}: {T['traj'].value_counts().to_dict()}")
    return T


TRAJ = {"primary: CDR 0.5, MMSE>=24, not demented; event = clinician-diagnosed dementia": ("cdr05_notdem", lambda d: d["demented"]),
        "sensitivity: event = CDR global >= 1": ("cdr05_notdem", lambda d: d["CDR"] >= 1),
        "sensitivity: clinician MCI baseline; event = clinician-diagnosed dementia": ("mci_clin", lambda d: d["demented"])}
TT = {}
for label, (bcol, ev) in TRAJ.items():
    log(" ", label); T = trajectory_cohort(bcol, ev); TT[label] = T
    if T["traj"].nunique() == 3 and len(T) >= 60 and enough(T["traj"].values, ["fast", "slow", "stable"]):
        PT, clsT = frozen_proba("trajectory", T)
        add("Aim3 trajectory (frozen ADNI portable model)", label, "portable", T["traj"].values, PT, clsT)
        for kc, c in enumerate(clsT):
            a1 = bin_auc(T["traj"].values == c, PT[:, kc]); log(f"    {c} vs rest AUC {a1:.3f}")
            ROWS.append(dict(analysis="Aim3 trajectory one-vs-rest", subset=f"{label}: {c} vs rest", model="portable", n=len(T), auc=round(a1, 3)))
        PTr = cv_proba(T[PORT], T["traj"].values, clsT)
        add("Aim3 trajectory (refitted in NACC, 5-fold CV)", label, "portable (refit)", T["traj"].values, PTr, clsT)
T0 = TT[list(TRAJ)[0]]
if len(F) and len(T0) and T0["traj"].nunique() == 3:
    fs_increment(T0, T0["traj"].values, sorted(T0["traj"].unique()), "trajectory", "nacc_fs_deltas_trajectory.csv")

# ============================ Aim 4: four groups ============================
log("\n================ Aim 4: frozen ADNI four-group model ================")
log("ADNI reference:", FROZEN["models"]["fourgroup"]["adni_cv_auc"], FROZEN["models"]["fourgroup"]["adni_cv_auc_ci"])
first = uds.drop_duplicates("sid").copy(); first["aid"] = np.arange(len(first))
mmF = nearest(first, uds, ["MMSE"], require="MMSE")
first = first.drop(columns=["MMSE"]).join(mmF[["MMSE"]], on="aid")
first["grp"] = np.where((num(first["NACCUDSD"]) == 1) & (first["CDR"] == 0), "CN",
                        np.where(first["demented"] & first["ad_primary"] & ((first["CDR"] >= 1) | ((first["CDR"] == 0.5) & (first["MMSE"] < 24))), "AD", None))
G = first[first["grp"].notna() & ~first["sid"].isin(set(dA["sid"]))]
M4 = dA.copy(); M4["grp"] = np.where(yA == "A+", "A+MCI", "A-MCI")
G4 = pd.concat([G[PORT + ["grp"]], M4[PORT + ["grp"]]], ignore_index=True)
log(f"  four-group counts: {G4['grp'].value_counts().to_dict()}")
if G4["grp"].nunique() == 4:
    P4, cls4 = frozen_proba("fourgroup", G4)
    add("Aim4 four-group (frozen ADNI portable model)", "CN / A-MCI / A+MCI / AD", "portable", G4["grp"].values, P4, cls4)
    for kc, c in enumerate(cls4):
        a1 = bin_auc(G4["grp"].values == c, P4[:, kc]); log(f"    {c} vs rest AUC {a1:.3f}")
        ROWS.append(dict(analysis="Aim4 four-group one-vs-rest", subset=f"{c} vs rest", model="portable", n=len(G4), auc=round(a1, 3)))

# ============================ Aim 5: autopsy ============================
log("\n================ Aim 5: autopsy (frozen amyloid model) ================")
npd = uds.groupby("sid")[["NACCNEUR", "NPADNC", "NPTHAL"]].last()
npd["plaques"] = valid(npd["NACCNEUR"], 0, 3); npd["adnc"] = valid(npd["NPADNC"], 0, 3)
has_np = npd[npd["plaques"].notna() | npd["adnc"].notna()]
log(f"  participants with neuropathology: {len(has_np)}")
mci_vis = uds[uds["cdr05_notdem"] & uds["sid"].isin(has_np.index)].copy(); mci_vis["aid"] = np.arange(len(mci_vis))
mmv = nearest(mci_vis, uds, ["MMSE"], require="MMSE")
mci_vis = mci_vis.drop(columns=["MMSE"]).join(mmv[["MMSE"]], on="aid")
mci_vis = mci_vis[mci_vis["MMSE"] >= 24].sort_values(["sid", "date"]).drop_duplicates("sid", keep="last")   # last MCI visit before death
Au = mci_vis.join(has_np[["plaques", "adnc"]], on="sid")
log(f"  autopsied participants with an MCI visit: {len(Au)}")
yod = valid(uds.groupby("sid")["NACCYOD"].last(), 1900, 2100); mod = valid(uds.groupby("sid")["NACCMOD"].last(), 1, 12)
Au["death"] = pd.to_datetime(dict(year=yod.reindex(Au["sid"]).values, month=mod.reindex(Au["sid"]).values, day=15), errors="coerce").values
Au["yrs_to_death"] = (Au["death"] - Au["date"]).dt.days / 365.25
for ref, lab in (("plaques", "moderate/frequent CERAD neuritic plaques"), ("adnc", "intermediate/high NIA-AA ADNC")):
    d = Au[Au[ref].notna()]
    log(f"    {lab}: n={len(d)}; median years from last MCI visit to death {d['yrs_to_death'].median():.1f}")
    add("Aim5 autopsy (frozen ADNI portable model)", f"last MCI visit; reference: {lab}", "portable",
        np.where(d[ref] >= 2, "A+", "A-"), amy_proba(d), CLS)
    d5 = d[d["yrs_to_death"].between(-0.1, 5)]
    add("Aim5 autopsy (frozen ADNI portable model)", f"last MCI visit <=5 years before death; reference: {lab}", "portable",
        np.where(d5[ref] >= 2, "A+", "A-"), amy_proba(d5), CLS)

# ============================ save ============================
pd.DataFrame(ROWS).to_csv(os.path.join(OUT, "nacc_results.csv"), index=False)
log("\nSaved aggregate results to", OUT)
LOGF.close()
