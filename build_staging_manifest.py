"""fixed_deep_manifest_staging.csv = fixed_deep_manifest.csv + within-MCI stage labels
(amyloid_mci: A+/A-, traj3: stable/slow/fast) for the imaged cohort. CPU/pandas."""
import pandas as pd, numpy as np, os
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); D=os.path.join(ROOT,"data")
man=pd.read_csv(os.path.join(HERE,"fixed_deep_manifest.csv"))
m=pd.read_csv(os.path.join(D,"master_features.csv"),low_memory=False)
b=pd.read_csv(os.path.join(D,"biomarkers_baseline.csv"))
conv=pd.read_csv(os.path.join(D,"cohort_conversion.csv"),low_memory=False)
X=m[["RID","baseline_dx"]].merge(b,on="RID",how="left")
def amy(r):
    if pd.notna(r.get("AMYPET_CENTILOID")): return "A+" if r["AMYPET_CENTILOID"]>=20 else "A-"
    if pd.notna(r.get("CSF_ABETA42")): return "A+" if r["CSF_ABETA42"]<980 else "A-"
    return np.nan
X["amyloid_mci"]=[amy(r) if r["baseline_dx"]=="MCI" else np.nan for _,r in X.iterrows()]
cv=conv[["RID","first_AD_month","max_followup_m"]].copy()
def traj(r):
    fa,fu=r["first_AD_month"],r["max_followup_m"]
    if pd.notna(fa): return "fast" if fa<=24 else ("slow" if fa<=48 else np.nan)
    return "stable" if (pd.notna(fu) and fu>=48) else np.nan
cv["traj3"]=cv.apply(traj,axis=1)
out=man.merge(X[["RID","amyloid_mci"]],on="RID",how="left").merge(cv[["RID","traj3"]],on="RID",how="left")
out.to_csv(os.path.join(HERE,"fixed_deep_manifest_staging.csv"),index=False)
print("staging manifest:",len(out),"subjects")
print("  amyloid_mci:",out["amyloid_mci"].value_counts(dropna=False).to_dict())
print("  traj3      :",out["traj3"].value_counts(dropna=False).to_dict())
