"""revision_data.py
Auxiliary subject-level tables for the revision analyses (reviewer analyses).
Reads the local ADNIMERGE2 .rda files and writes to data/:
  amyloid_detail_baseline.csv : the SAME baseline amyloid-PET and CSF records used for
      the primary label (identical selection rule to extract_biomarkers), plus scan/
      exam date, PET tracer, tracer-specific UC Berkeley status, Abeta40, p-tau, t-tau.
  neuropath.csv               : autopsy measures (Thal phase, CERAD, ADNC, co-pathologies).
  baseline_dx_detail.csv      : baseline dementia/MCI etiology fields from DXSUM + site.
"""
import os, re, warnings, numpy as np, pandas as pd, pyreadr
warnings.filterwarnings("ignore")
PROJ=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT=os.path.dirname(PROJ)
RDA=os.path.join(ROOT,"adni images","ADNIMERGE2","data"); OUT=os.path.join(PROJ,"data")
def load(n): r=pyreadr.read_r(os.path.join(RDA,n+".rda")); return r[list(r.keys())[0]]
def vmonth(v):
    v=str(v).lower().strip()
    if v in ("bl","sc","scmri","init","blmri","m0","v01","v02"): return 0
    mm=re.match(r"m(\d+)",v); return int(mm.group(1)) if mm else 999
def first_row(df,key,keep):
    """Same ordering rule as extract_biomarkers: sort by visit month, then the first
    record with a non-missing `key` value (== groupby().first() for that column)."""
    df=df.copy(); df["RID"]=pd.to_numeric(df["RID"],errors="coerce"); df=df.dropna(subset=["RID"])
    df["RID"]=df["RID"].astype(int); df["_m"]=df["VISCODE2"].map(vmonth); df=df.sort_values("_m")
    df[key]=pd.to_numeric(df[key],errors="coerce"); df=df[df[key].notna()]
    return df.groupby("RID",sort=True).head(1)[["RID"]+keep]

amy=load("UCBERKELEY_AMY_6MM")
for c in ["CENTILOIDS","SUMMARY_SUVR","AMYLOID_STATUS","AMYLOID_STATUS_COMPOSITE_REF","qc_flag"]:
    amy[c]=pd.to_numeric(amy[c],errors="coerce")
pet=first_row(amy,"CENTILOIDS",["SCANDATE","TRACER","CENTILOIDS","SUMMARY_SUVR","AMYLOID_STATUS",
                                "AMYLOID_STATUS_COMPOSITE_REF","qc_flag","VISCODE2"])
pet=pet.rename(columns={"SCANDATE":"PET_DATE","TRACER":"PET_TRACER","CENTILOIDS":"PET_CL",
                        "SUMMARY_SUVR":"PET_SUVR","AMYLOID_STATUS":"PET_STATUS_UCB",
                        "AMYLOID_STATUS_COMPOSITE_REF":"PET_STATUS_UCB_COMPREF","qc_flag":"PET_QC",
                        "VISCODE2":"PET_VISCODE2"})
csf=load("UPENNBIOMK_ROCHE_ELECSYS")
for c in ["ABETA40","ABETA42","TAU","PTAU"]: csf[c]=pd.to_numeric(csf[c],errors="coerce")
csf=first_row(csf,"ABETA42",["EXAMDATE","ABETA42","ABETA40","PTAU","TAU","VISCODE2"])
csf=csf.rename(columns={"EXAMDATE":"CSF_DATE","ABETA42":"CSF_AB42","ABETA40":"CSF_AB40",
                        "PTAU":"CSF_PTAU181","TAU":"CSF_TTAU","VISCODE2":"CSF_VISCODE2"})
det=pet.merge(csf,on="RID",how="outer")
det.to_csv(os.path.join(OUT,"amyloid_detail_baseline.csv"),index=False)
print("amyloid_detail_baseline:",det.shape)

npth=load("NEUROPATH")
keep=["RID","NPDAGE","NPDODYR","NPTHAL","NPBRAAK","NPNEUR","NPADNC","NPAMY","NPLBOD","NPTDPB",
      "NPTDPC","NPTDPD","NPTDPE","NPHIPSCL","NPINF","NPFTDTAU","NPPATH","NPARTER","NPWMR"]
npth=npth[[c for c in keep if c in npth.columns]].copy()
npth["RID"]=pd.to_numeric(npth["RID"],errors="coerce"); npth=npth.dropna(subset=["RID"])
npth["RID"]=npth["RID"].astype(int)
for c in npth.columns:
    if c!="RID": npth[c]=pd.to_numeric(npth[c],errors="coerce")
npth.to_csv(os.path.join(OUT,"neuropath.csv"),index=False); print("neuropath:",npth.shape)

dx=load("DXSUM")[["RID","PTID","SITEID","VISCODE2","EXAMDATE","DIAGNOSIS","DXDDUE","DXMDUE","DXODES","DXMOTHET"]].copy()
dx["RID"]=pd.to_numeric(dx["RID"],errors="coerce"); dx=dx.dropna(subset=["RID"]); dx["RID"]=dx["RID"].astype(int)
dx["EXAMDATE"]=pd.to_datetime(dx["EXAMDATE"],errors="coerce")
dx.to_csv(os.path.join(OUT,"dxsum_detail.csv"),index=False); print("dxsum_detail:",dx.shape)
