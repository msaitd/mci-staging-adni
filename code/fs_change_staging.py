"""28_fs_change_staging.py
Leakage-safe FreeSurfer-CHANGE arm for within-MCI staging (Paper B).

Design:
  * Landmark L = 12 months. Annualized slope of hippocampus (HIPP), ventricle
    (VENT) and entorhinal thickness (ENT) computed from FreeSurfer visits in
    [0, L] ONLY (past data relative to the landmark).
  * Prognostic (trajectory) incremental-value test: among MCI NOT converted by
    month L, does clinical + early-atrophy-rate beat clinical alone at predicting
    the stable/slow/fast trajectory (events strictly after L)?  -> leakage-safe.
  * Biological association: is early (0-12 mo) atrophy rate faster in amyloid-
    positive MCI?  Reported as group means + t-test (characterization, not a
    triage predictor; amyloid status is a baseline label).
Subject-level CV via ml_common; logistic primary + 3 models.
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ml_common import evaluate_combo
from sklearn.metrics import roc_auc_score
from scipy import stats

HERE=os.path.dirname(__file__); ROOT=os.path.dirname(HERE)
D=os.path.join(ROOT,"data"); OUT=os.path.join(ROOT,"outputs","within_mci")
os.makedirs(OUT,exist_ok=True)
L_DAYS=365  # landmark window

# ---- FreeSurfer per-visit ROIs (ICV-normalized), same definitions as code/08 ----
fs=pd.read_csv(os.path.join(D,"freesurfer_fsx7.csv"),low_memory=False)
fs["EXAMDATE"]=pd.to_datetime(fs["EXAMDATE"],errors="coerce"); fs=fs.dropna(subset=["EXAMDATE"])
icv=pd.to_numeric(fs["ST10CV"],errors="coerce")
fs["HIPP"]=(pd.to_numeric(fs["ST29SV"],errors="coerce")+pd.to_numeric(fs["ST88SV"],errors="coerce"))/icv*1000
fs["VENT"]=(pd.to_numeric(fs["ST37SV"],errors="coerce")+pd.to_numeric(fs["ST96SV"],errors="coerce"))/icv*1000
fs["ENT"] =(pd.to_numeric(fs["ST24TA"],errors="coerce")+pd.to_numeric(fs["ST83TA"],errors="coerce"))/2

def landmark_slopes(win_days):
    rows=[]
    for rid,g in fs.groupby("RID"):
        g=g.sort_values("EXAMDATE"); t0=g["EXAMDATE"].min()
        yrs=(g["EXAMDATE"]-t0).dt.days.values/365.25
        mask=(g["EXAMDATE"]-t0).dt.days.values<=win_days
        rec={"RID":rid}
        for m in ["HIPP","VENT","ENT"]:
            v=pd.to_numeric(g[m],errors="coerce").values
            ok=mask & np.isfinite(v) & np.isfinite(yrs)
            if ok.sum()>=2 and np.ptp(yrs[ok])>0.2:
                rec[f"fsch_{m}"]=np.polyfit(yrs[ok],v[ok],1)[0]
            else:
                rec[f"fsch_{m}"]=np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

sl=landmark_slopes(L_DAYS)
ch_cols=["fsch_HIPP","fsch_VENT","fsch_ENT"]
sl_ok=sl.dropna(subset=ch_cols, how="any")

# ---- labels ----
m=pd.read_csv(os.path.join(D,"master_features.csv"),low_memory=False)
b=pd.read_csv(os.path.join(D,"biomarkers_baseline.csv"))
conv=pd.read_csv(os.path.join(D,"cohort_conversion.csv"),low_memory=False)
fam=json.load(open(os.path.join(D,"feature_families.json")))
demo,cog=fam["demo"],fam["cognition"]
X=m.merge(b,on="RID",how="left")
def amy(r):
    if pd.notna(r.get("AMYPET_CENTILOID")): return 1.0 if r["AMYPET_CENTILOID"]>=20 else 0.0
    if pd.notna(r.get("CSF_ABETA42")): return 1.0 if r["CSF_ABETA42"]<980 else 0.0
    return np.nan
X["A"]=X.apply(amy,axis=1)
cvv=conv[["RID","first_AD_month","max_followup_m"]].copy()
def traj(r):
    fa,fu=r["first_AD_month"],r["max_followup_m"]
    if pd.notna(fa): return "fast" if fa<=24 else ("slow" if fa<=48 else np.nan)
    return "stable" if (pd.notna(fu) and fu>=48) else np.nan
cvv["traj"]=cvv.apply(traj,axis=1)
X=X.merge(cvv,on="RID",how="left").merge(sl,on="RID",how="left")

MODELS=["logistic_l2","extra_trees","hist_gb"]
def run(sub,label_col,classes,cols,tag):
    Xd=sub[cols].apply(pd.to_numeric,errors="coerce")
    Xd=Xd.fillna(Xd.median(numeric_only=True))
    y=sub[label_col].astype(str).values; g=sub["RID"].values
    best=None
    for mm in MODELS:
        _,oof=evaluate_combo(Xd,y,g,classes,mm)
        if len(classes)==2:
            yt=(oof["y_true"]==classes[1]).astype(int).values
            auc=roc_auc_score(yt,oof["p_"+classes[1]].values)
        else:
            from sklearn.preprocessing import label_binarize
            yt=label_binarize(oof["y_true"].astype(str),classes=classes)
            P=oof[["p_"+c for c in classes]].values
            auc=roc_auc_score(yt,P,average="macro",multi_class="ovr")
        row=(tag,mm,len(sub),round(auc,3))
        if best is None or auc>best[3]: best=row
        print("   ",row)
    return best

print("=== Landmark FS-change availability ===")
print("subjects with 0-12mo FS slope (>=2 visits):",len(sl_ok))

# ---------- (A) TRAJECTORY incremental value (leakage-safe landmark) ----------
# exclude subjects converted by <=12 mo so outcome is strictly after the landmark
tr=X[X.traj.notna() & X[ch_cols].notna().all(axis=1)].copy()
tr=tr[~(tr.first_AD_month<=12)]
classes=["stable","slow","fast"]
print("\n=== TRAJECTORY (landmark, converted>12mo only) n=",len(tr),tr.traj.value_counts().to_dict(),"===")
res=[]
res.append(run(tr,"traj",classes,[c for c in demo+cog if c in tr],"clinical"))
res.append(run(tr,"traj",classes,ch_cols,"fs_change_only"))
res.append(run(tr,"traj",classes,[c for c in demo+cog if c in tr]+ch_cols,"clinical+fs_change"))

# ---------- (B) AMYLOID association: early atrophy rate by amyloid status ----------
am=X[X.baseline_dx.eq("MCI") & X.A.notna() & X[ch_cols].notna().all(axis=1)].copy()
print("\n=== AMYLOID early-atrophy association n=",len(am),"(A+=",int((am.A==1).sum()),"A-=",int((am.A==0).sum()),") ===")
assoc=[]
for m_ in ch_cols:
    ap=am[am.A==1][m_].values; an=am[am.A==0][m_].values
    t,p=stats.ttest_ind(ap,an,equal_var=False)
    assoc.append((m_,round(np.nanmean(ap),4),round(np.nanmean(an),4),round(float(t),2),float(p)))
    print(f"   {m_}: A+ mean={np.nanmean(ap):.4f}  A- mean={np.nanmean(an):.4f}  t={t:.2f}  p={p:.2e}")

# ---------- save ----------
pd.DataFrame(res,columns=["featureset","best_model","n","AUC"]).to_csv(
    os.path.join(OUT,"fs_change_trajectory.csv"),index=False)
pd.DataFrame(assoc,columns=["measure","A_pos_mean_annual","A_neg_mean_annual","t","p"]).to_csv(
    os.path.join(OUT,"fs_change_amyloid_assoc.csv"),index=False)
print("\nsaved fs_change_trajectory.csv & fs_change_amyloid_assoc.csv")
