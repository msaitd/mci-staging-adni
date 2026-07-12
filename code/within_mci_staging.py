"""Within-MCI staging: (A) amyloid A+/A- (biological), (B) conversion trajectory
stable/slow/fast (prognostic), (C) ordinal continuum CN<A-MCI<A+MCI<AD.
Leakage-safe subject-level CV via ml_common. Amyloid task uses NO molecular
markers as predictors (would be circular). Resumable."""
import os, sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ml_common import make_models, evaluate_combo
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize

HERE=os.path.dirname(__file__); ROOT=os.path.dirname(HERE)
D=os.path.join(ROOT,"data"); OUT=os.path.join(ROOT,"outputs","within_mci")
os.makedirs(OUT,exist_ok=True)
SUMMARY=os.path.join(OUT,"within_mci_summary_v2.csv")
BUDGET=28.0; t0=time.time()

m=pd.read_csv(os.path.join(D,"master_features.csv"),low_memory=False)
b=pd.read_csv(os.path.join(D,"biomarkers_baseline.csv"))
conv=pd.read_csv(os.path.join(D,"cohort_conversion.csv"),low_memory=False)
fam=json.load(open(os.path.join(D,"feature_families.json")))
demo,cog,fs=fam["demo"],fam["cognition"],fam["freesurfer"]
bio=[c for c in b.columns if c!="RID"]
X=m.merge(b,on="RID",how="left")

# amyloid positivity: Centiloid>=20 (primary) else CSF Abeta42<980
def amy(r):
    if pd.notna(r.get("AMYPET_CENTILOID")): return 1.0 if r["AMYPET_CENTILOID"]>=20 else 0.0
    if pd.notna(r.get("CSF_ABETA42")): return 1.0 if r["CSF_ABETA42"]<980 else 0.0
    return np.nan
X["A"]=X.apply(amy,axis=1)
# trajectory
cv=conv[["RID","first_AD_month","max_followup_m"]].copy()
def traj(r):
    fa,fu=r["first_AD_month"],r["max_followup_m"]
    if pd.notna(fa): return "fast" if fa<=24 else ("slow" if fa<=48 else np.nan)
    return "stable" if (pd.notna(fu) and fu>=48) else np.nan
cv["traj"]=cv.apply(traj,axis=1)
X=X.merge(cv[["RID","traj"]],on="RID",how="left")
# ordinal level
def ordlvl(r):
    dx=r["baseline_dx"]
    if dx=="CN": return "CN"
    if dx=="AD": return "AD"
    if dx=="MCI": return "A-MCI" if r["A"]==0 else ("A+MCI" if r["A"]==1 else np.nan)
    return np.nan
X["ordlvl"]=X.apply(ordlvl,axis=1)

# task defs: (name, row_mask, label_col, classes, featuresets)
def fsets(include_bio):
    d={"demo_apoe":demo,"cognition":cog,"freesurfer":fs,"clinical":demo+cog,"clinical+fs":demo+cog+fs}
    if include_bio: d["all"]=demo+cog+fs+bio
    return d
TASKS={
 "amyloid_mci": (X["baseline_dx"].eq("MCI") & X["A"].notna(), "A", ["A-","A+"],
                 {"demo_apoe":demo,"cognition":cog,"freesurfer":fs,"clinical":demo+cog,"clinical+fs":demo+cog+fs}),
 "trajectory":  (X["traj"].notna(), "traj", ["stable","slow","fast"], fsets(True)),
 "ordinal":     (X["ordlvl"].notna(),"ordlvl",["CN","A-MCI","A+MCI","AD"],
                 {"demo_apoe":demo,"cognition":cog,"freesurfer":fs,"clinical":demo+cog,"clinical+fs":demo+cog+fs}),
}
MODELS=["logistic_l2","extra_trees","hist_gb"]

def lab(v,col):  # map label col to class strings
    if col=="A": return np.where(v==1.0,"A+","A-")
    return v.astype(str)

def metrics_from_oof(oof, classes):
    y=oof["y_true"].astype(str).values
    P=oof[["p_"+c for c in classes]].values
    yp=oof["y_pred"].astype(str).values
    bacc=balanced_accuracy_score(y,yp)
    if len(classes)==2:
        yb=(y==classes[1]).astype(int); auc=roc_auc_score(yb,P[:,1]); ap=average_precision_score(yb,P[:,1])
    else:
        Y=label_binarize(y,classes=classes); auc=roc_auc_score(Y,P,average="macro"); ap=np.nan
    # bootstrap AUC CI
    rng=np.random.default_rng(42); aucs=[]
    for _ in range(100):
        idx=rng.integers(0,len(y),len(y))
        try:
            if len(classes)==2: aucs.append(roc_auc_score((y[idx]==classes[1]).astype(int),P[idx,1]))
            else: aucs.append(roc_auc_score(label_binarize(y[idx],classes=classes),P[idx],average="macro"))
        except Exception: pass
    lo,hi=(np.percentile(aucs,[2.5,97.5]) if aucs else (np.nan,np.nan))
    return bacc,auc,ap,lo,hi

done=set()
if os.path.exists(SUMMARY):
    old=pd.read_csv(SUMMARY)
    done={(r.task,r.featureset,r.model) for r in old.itertuples()}
rows=[]
for tname,(mask,col,classes,FS) in TASKS.items():
    sub=X[mask].copy()
    y=lab(sub[col].values,col)
    groups=sub["RID"].values
    for fsname,cols in FS.items():
        cols=[c for c in cols if c in sub.columns]
        for mdl in MODELS:
            if tname=="ordinal" and fsname in ("freesurfer","clinical+fs") and mdl=="extra_trees": continue
            if (tname,fsname,mdl) in done: continue
            if time.time()-t0>BUDGET:
                if rows:
                    df=pd.DataFrame(rows)
                    hdr=not os.path.exists(SUMMARY)
                    df.to_csv(SUMMARY,mode="a",header=hdr,index=False)
                print("TIME budget; wrote",len(rows),"rows; RESUME"); sys.exit(0)
            Xsub=sub[cols]
            folds,oof=evaluate_combo(Xsub,y,groups,classes,mdl,n_splits=5,n_repeats=1,seed=42)
            bacc,auc,ap,lo,hi=metrics_from_oof(oof,classes)
            oof.to_csv(os.path.join(OUT,f"oof_{tname}_{fsname}_{mdl}.csv"),index=False)
            row=dict(task=tname,featureset=fsname,model=mdl,n=len(sub),n_classes=len(classes),
                     bAcc=round(bacc,4),AUC=round(auc,4),AUPRC=(round(ap,4) if ap==ap else ""),
                     AUC_lo=round(lo,4),AUC_hi=round(hi,4))
            hdr=not os.path.exists(SUMMARY)
            pd.DataFrame([row]).to_csv(SUMMARY,mode="a",header=hdr,index=False)  # ANINDA yaz
            done.add((tname,fsname,mdl)); rows.append(row)
            print(f"{tname:12} {fsname:12} {mdl:12} n={len(sub)} bAcc={bacc:.3f} AUC={auc:.3f} [{lo:.3f}-{hi:.3f}]",flush=True)
if rows:
    df=pd.DataFrame(rows); hdr=not os.path.exists(SUMMARY)
    df.to_csv(SUMMARY,mode="a",header=hdr,index=False)
print("DONE_ALL" if rows else "NOTHING_NEW", "| elapsed",round(time.time()-t0,1))
