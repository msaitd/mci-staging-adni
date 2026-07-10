"""29_amyloid_utility.py
Clinical-utility & rigor add-ons for the amyloid A+/A- MCI PRIMARY model (Paper B):
  * Calibration (Brier, calibration intercept/slope, decile reliability)
  * Decision-curve analysis (net benefit vs treat-all / treat-none)
  * Operating point at ~90% sensitivity (triage framing)
  * APOE-stratified discrimination (carriers vs non-carriers)
  * Formal DeLong-style paired bootstrap DAUC (clinical vs clinical+FS, vs FS)
  * Baseline cohort characteristics by amyloid status
All from out-of-fold logistic probabilities; subject-level; no leakage.
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ml_common import evaluate_combo
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats

HERE=os.path.dirname(__file__); ROOT=os.path.dirname(HERE)
D=os.path.join(ROOT,"data"); OUT=os.path.join(ROOT,"outputs","within_mci")
m=pd.read_csv(os.path.join(D,"master_features.csv"),low_memory=False)
b=pd.read_csv(os.path.join(D,"biomarkers_baseline.csv"))
fam=json.load(open(os.path.join(D,"feature_families.json")))
demo,cog,fs=fam["demo"],fam["cognition"],fam["freesurfer"]
X=m.merge(b,on="RID",how="left")
def amy(r):
    if pd.notna(r.get("AMYPET_CENTILOID")): return 1.0 if r["AMYPET_CENTILOID"]>=20 else 0.0
    if pd.notna(r.get("CSF_ABETA42")): return 1.0 if r["CSF_ABETA42"]<980 else 0.0
    return np.nan
X["A"]=X.apply(amy,axis=1)
sub=X[X.baseline_dx.eq("MCI") & X.A.notna()].copy()
y=np.where(sub["A"].values==1.0,"A+","A-"); g=sub["RID"].values
classes=["A-","A+"]
def oof_prob(cols):
    Xd=sub[cols].apply(pd.to_numeric,errors="coerce"); Xd=Xd.fillna(Xd.median(numeric_only=True))
    _,oof=evaluate_combo(Xd,y,g,classes,"logistic_l2")
    return oof[["RID","y_true","p_A+"]].rename(columns={"p_A+":"p"})
clin=oof_prob(demo+cog); clfs=oof_prob(demo+cog+fs); fso=oof_prob(fs)
base=clin.merge(sub[["RID","APOE4"]],on="RID",how="left")
yt=(base.y_true=="A+").astype(int).values; p=base.p.values
prev=yt.mean(); N=len(yt)
print(f"n={N} prevalence(A+)={prev:.3f}")

# ---- calibration ----
eps=1e-6; pc=np.clip(p,eps,1-eps); logit=np.log(pc/(1-pc))
sl,ic=np.polyfit(logit,yt,1)  # crude; use logistic below
import numpy as _np
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression().fit(logit.reshape(-1,1),yt)
cal_slope=float(lr.coef_[0,0]); cal_int=float(lr.intercept_[0])
brier=brier_score_loss(yt,p)
print(f"Brier={brier:.3f} calibration slope={cal_slope:.2f} intercept={cal_int:.2f}")
# decile reliability
dec=pd.qcut(p,10,duplicates="drop")
rel=pd.DataFrame({"p":p,"y":yt,"bin":dec}).groupby("bin",observed=True).agg(pred=("p","mean"),obs=("y","mean"),n=("y","size")).reset_index(drop=True)

# ---- decision curve (net benefit) ----
def net_benefit(th):
    yhat=(p>=th).astype(int); tp=((yhat==1)&(yt==1)).sum(); fp=((yhat==1)&(yt==0)).sum()
    nb=tp/N - fp/N*(th/(1-th)); nb_all=prev-(1-prev)*(th/(1-th)); return nb,nb_all
ths=[0.1,0.2,0.3,0.4,0.5]; dca=[(t,)+net_benefit(t)+ (0.0,) for t in ths]
dca=pd.DataFrame([(t,round(nb,3),round(na,3),0.0) for t,(nb,na) in [(t,net_benefit(t)) for t in ths]],
                 columns=["threshold","NB_model","NB_treat_all","NB_treat_none"])

# ---- operating point ~90% sensitivity ----
order=np.argsort(-p)
best=None
for t in np.unique(p):
    yhat=(p>=t).astype(int); tp=((yhat==1)&(yt==1)).sum(); fn=((yhat==0)&(yt==1)).sum()
    fp=((yhat==1)&(yt==0)).sum(); tn=((yhat==0)&(yt==0)).sum()
    sens=tp/(tp+fn);
    if sens>=0.90:
        spec=tn/(tn+fp); ppv=tp/(tp+fp) if (tp+fp) else np.nan; npv=tn/(tn+fn) if (tn+fn) else np.nan
        referred=(yhat==1).mean()
        best=dict(threshold=round(float(t),3),sensitivity=round(sens,3),specificity=round(spec,3),
                  ppv=round(ppv,3),npv=round(npv,3),referred_frac=round(float(referred),3))
print("Operating point (>=90% sens):",best)

# ---- APOE-stratified AUC (clinical model) ----
strat={}
for lab,mask in [("APOE4_noncarrier",base.APOE4==0),("APOE4_carrier",base.APOE4>=1)]:
    mk=mask.values & np.isfinite(base.APOE4.values)
    if mk.sum()>20 and len(np.unique(yt[mk]))==2:
        strat[lab]=(int(mk.sum()),round(roc_auc_score(yt[mk],p[mk]),3),round(yt[mk].mean(),3))
print("APOE-stratified clinical AUC:",strat)

# ---- paired bootstrap DAUC ----
def dauc(pa,pb,n=2000,seed=1):
    rng=np.random.default_rng(seed); diffs=[]
    idx=np.arange(N)
    for _ in range(n):
        s=rng.choice(idx,N,replace=True)
        if len(np.unique(yt[s]))<2: continue
        diffs.append(roc_auc_score(yt[s],pa[s])-roc_auc_score(yt[s],pb[s]))
    diffs=np.array(diffs); lo,hi=np.percentile(diffs,[2.5,97.5])
    pval=2*min((diffs<=0).mean(),(diffs>=0).mean())
    return round(float(diffs.mean()),3),round(float(lo),3),round(float(hi),3),round(float(pval),3)
pcl=base.p.values
pcf=clfs.set_index("RID").loc[base.RID,"p"].values
pfs=fso.set_index("RID").loc[base.RID,"p"].values
d1=dauc(pcl,pcf); d2=dauc(pcl,pfs)
print(f"DAUC clinical-vs-(clinical+FS): {d1}")
print(f"DAUC clinical-vs-FreeSurfer:    {d2}")

# ---- cohort characteristics by amyloid status ----
def desc(col,cont=True):
    ap=pd.to_numeric(sub[sub.A==1][col],errors="coerce"); an=pd.to_numeric(sub[sub.A==0][col],errors="coerce")
    if cont:
        t,pp=stats.ttest_ind(ap,an,nan_policy="omit",equal_var=False)
        return f"{ap.mean():.1f}±{ap.std():.1f}", f"{an.mean():.1f}±{an.std():.1f}", f"{pp:.1e}"
    else:
        ap2=(ap>=1).mean()*100; an2=(an>=1).mean()*100
        return f"{ap2:.0f}%", f"{an2:.0f}%","-"
rows=[]
rows.append(("N",str(int((sub.A==1).sum())),str(int((sub.A==0).sum())),"-"))
for c,lab,cont in [("age","Age, y",True),("sex_female","Female",False),("PTEDUCAT","Education, y",True),
                   ("APOE4","APOE e4 carrier",False),("MMSE","MMSE",True),("CDRSB","CDR-SB",True),("ADAS13","ADAS-Cog13",True)]:
    if c in sub: rows.append((lab,)+desc(c,cont))
chars=pd.DataFrame(rows,columns=["Characteristic","A+ MCI","A- MCI","p"])
print("\n",chars.to_string(index=False))

# ---- save ----
rel.to_csv(os.path.join(OUT,"amyloid_calibration_deciles.csv"),index=False)
dca.to_csv(os.path.join(OUT,"amyloid_dca.csv"),index=False)
chars.to_csv(os.path.join(OUT,"amyloid_cohort_characteristics.csv"),index=False)
pd.DataFrame([dict(metric="brier",v=round(brier,3)),dict(metric="cal_slope",v=round(cal_slope,2)),
             dict(metric="cal_intercept",v=round(cal_int,2))]).to_csv(os.path.join(OUT,"amyloid_calibration_summary.csv"),index=False)
print("\nsaved utility outputs")
