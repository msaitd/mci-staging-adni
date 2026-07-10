"""
build_deep_manifest.py  (run on your machine, normal Python)
Scan CAT12 outputs, pair mwp1+mwp2 per scan, map to subject (RID) and to the
expanded-study labels (baseline_dx, conv_36). Output: deep_manifest.csv used by
the GPU training script. One baseline scan per subject.
"""
import os, re, glob
import pandas as pd

# ----------------------------- CONFIG --------------------------------
DERIV = r"F:\ADNI_derivatives\cat12"     # must match step1_cat12_segment.m
# ---------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

def parse(fp):
    p = re.search(r"(\d{3}_S_\d{4})", fp)
    i = re.search(r"_I(\d+)\.nii", fp)
    return (p.group(1) if p else None, int(i.group(1)) if i else None)

rows = {}
for kind in ("mwp1", "mwp2"):
    for fp in glob.glob(os.path.join(DERIV, "**", kind + "*.nii"), recursive=True):
        ptid, iid = parse(fp)
        if ptid is None or iid is None:
            continue
        rows.setdefault((ptid, iid), {})[kind] = fp
recs = []
for (ptid, iid), d in rows.items():
    if "mwp1" in d and "mwp2" in d:
        recs.append(dict(PTID=ptid, ImageUID=iid, path_mwp1=d["mwp1"], path_mwp2=d["mwp2"]))
man = pd.DataFrame(recs)
if man.empty:
    raise SystemExit("No paired mwp1/mwp2 maps found under %s (run CAT12 first)." % DERIV)

# map PTID -> RID and attach labels
dx = pd.read_csv(os.path.join(DATA, "dxsum.csv"))
p2r = dict(zip(dx["PTID"].astype(str), dx["RID"]))
man["RID"] = man["PTID"].astype(str).map(p2r)
man = man.dropna(subset=["RID"]); man["RID"] = man["RID"].astype(int)

master = pd.read_csv(os.path.join(DATA, "master_features.csv"), low_memory=False)
lab = master[["RID", "baseline_dx", "conv_36"]].drop_duplicates("RID")
man = man.merge(lab, on="RID", how="left")

# one baseline scan per subject: prefer the baseline FreeSurfer ImageUID, else lowest
mani = pd.read_csv(os.path.join(DATA, "manifest_subject_baseline.csv"), low_memory=False)
base_iid = dict(zip(mani["RID"], mani["IMAGEUID"]))
def pick(g):
    bi = base_iid.get(g["RID"].iloc[0])
    if bi is not None and (g["ImageUID"] == bi).any():
        return g[g["ImageUID"] == bi].iloc[0]
    return g.sort_values("ImageUID").iloc[0]
man = man.groupby("RID", group_keys=False).apply(pick).reset_index(drop=True)

out = os.path.join(HERE, "deep_manifest.csv")
man.to_csv(out, index=False)
print("deep_manifest.csv: %d subjects" % len(man))
print("  baseline_dx:", man["baseline_dx"].value_counts(dropna=False).to_dict())
print("  conv_36    :", man["conv_36"].value_counts(dropna=False).to_dict())
print("saved ->", out)
