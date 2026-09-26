"""figure5_external_validation.py -- Figure 5 (external validation in OASIS-3), drawn only from aggregate results:
OASIS-3 aggregate outputs (external_validation/oasis_results, produced by oasis3_external_validation.py), the frozen-model
reference AUCs (frozen_adni_models.json) and the ADNI portable-imaging comparison (J_portable_imaging.csv)."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EV = os.path.join(ROOT, "external_validation")
RES = os.path.join(EV, "oasis_results")
FIG = os.path.join(ROOT, "figures", "revision")
plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold", "axes.titlesize": 9.5, "font.family": "DejaVu Sans"})

r = pd.read_csv(os.path.join(RES, "oasis_results.csv"))
fz = json.load(open(os.path.join(EV, "frozen_adni_models.json")))["models"]
J = pd.read_csv(os.path.join(ROOT, "outputs", "revision", "J_portable_imaging.csv"))
dec = pd.read_csv(os.path.join(RES, "oasis_calibration_deciles.csv"))
cal = pd.read_csv(os.path.join(RES, "oasis_calibration_summary.csv")).iloc[0]
dA = pd.read_csv(os.path.join(RES, "oasis_fs_deltas_amyloid.csv"))
dT = pd.read_csv(os.path.join(RES, "oasis_fs_deltas_trajectory.csv"))


def row(analysis_start, subset_start, model="portable"):
    k = r["analysis"].str.startswith(analysis_start) & r["subset"].str.startswith(subset_start) & (r["model"] == model)
    x = r[k].iloc[0]; return float(x.auc), float(x.ci_lo), float(x.ci_hi), int(x.n)


A1 = "Aim1 amyloid (frozen"
items = [("Amyloid status in MCI", None),
         ("ADNI, cross-validated (n=1,059)", ("ADNI", fz["amyloid"]["adni_cv_auc"], *fz["amyloid"]["adni_cv_auc_ci"])),
         ("OASIS-3 primary: PiB, CDR 0.5, MMSE ≥24", (A1, "primary cohort, Centiloid >= 20")),
         ("Centiloid ≥12", (A1, "primary cohort, Centiloid >= 12")),
         ("Centiloid ≥30", (A1, "primary cohort, Centiloid >= 30")),
         ("Partial-volume-corrected Centiloid", (A1, "primary cohort, PVC")),
         ("PET quality control passed", (A1, "PUP QC passed")),
         ("Clinical visit ≤90 days from PET", (A1, "clinical visit within 90")),
         ("Non-AD clinical aetiology excluded", (A1, "sensitivity: CDR 0.5, MMSE>=24, clinician non-AD")),
         ("Clinician label 'not demented'", (A1, "sensitivity: CDR 0.5, MMSE>=24, clinician label")),
         ("No MMSE criterion", (A1, "sensitivity: CDR 0.5, any MMSE")),
         ("Florbetapir PET", (A1, "AV45 only")),
         ("PiB and florbetapir pooled", (A1, "all tracers")),
         ("Stable/slow/fast trajectory", None),
         ("ADNI, cross-validated (n=612)", ("ADNI", fz["trajectory"]["adni_cv_auc"], *fz["trajectory"]["adni_cv_auc_ci"])),
         ("OASIS-3 primary: progression to CDR ≥1", ("Aim3 trajectory (frozen", "primary")),
         ("Clinician-diagnosed dementia also counted", ("Aim3 trajectory (frozen", "sensitivity")),
         ("Four groups (CN / A−MCI / A+MCI / AD)", None),
         ("ADNI, cross-validated (n=3,221)", ("ADNI", fz["fourgroup"]["adni_cv_auc"], *fz["fourgroup"]["adni_cv_auc_ci"])),
         ("OASIS-3", ("Aim4 four-group (frozen", "CN"))]

fig = plt.figure(figsize=(16.5, 7.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1], wspace=0.62, hspace=0.45)
ax = fig.add_subplot(gs[:, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[1, 1])

y = len(items); ticks, labels = [], []
for lab, spec in items:
    if spec is None:
        ax.text(0.515, y, lab, fontweight="bold", va="center", fontsize=8.8)
    else:
        if spec[0] == "ADNI":
            v, lo, hi = spec[1], spec[2], spec[3]; mk, mfc, col = "D", "white", "#7f7f7f"
        else:
            v, lo, hi, n = row(spec[0], spec[1]); mk, mfc, col = "o", "#2c5496", "#2c5496"
            if "n=" not in lab:
                lab = f"{lab} (n={n:,})"
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt=mk, color=col, mfc=mfc, ms=5.5, capsize=2.5, lw=1.1)
        ax.text(1.012, y, f"{v:.2f} ({lo:.2f}–{hi:.2f})", va="center", fontsize=7.6)
        ticks.append(y); labels.append(lab)
    y -= 1
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.8)
ax.set_xlim(0.5, 1.13); ax.set_xticks(np.arange(0.5, 1.01, 0.1)); ax.set_ylim(0.3, len(items) + 0.7)
ax.axvline(0.5, color="grey", lw=0.6, ls="--")
ax.set_xlabel("ROC AUC (95% bootstrap CI); macro one-vs-rest for multiclass")
ax.set_title("A  Frozen ADNI portable models in OASIS-3", loc="left")
ax.plot([], [], "D", color="#7f7f7f", mfc="white", label="ADNI (cross-validated reference)")
ax.plot([], [], "o", color="#2c5496", label="OASIS-3 (external, model applied unchanged)")
ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.07), ncol=2, fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=0.3)

axB.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8, label="Ideal")
axB.plot(dec["mean_pred"], dec["observed"], "o-", color="#2c5496", ms=5, label="Observed (deciles)")
axB.set_xlim(0, 1); axB.set_ylim(0, 1.02); axB.set_xlabel("Predicted probability (A+)"); axB.set_ylabel("Observed frequency")
axB.text(0.03, 0.97, f"Brier {cal.brier:.3f}\nslope {cal.cal_slope:.2f}\ncalibration-in-the-large {cal.cal_intercept_citl:.2f}",
         va="top", fontsize=7.8, bbox=dict(boxstyle="round", fc="white", ec="grey"))
axB.legend(loc="lower right", fontsize=7.8); axB.set_title("B  Calibration of the frozen amyloid model (OASIS-3, PiB, n=98)", loc="left")
axB.spines[["top", "right"]].set_visible(False)


def jrow(task, name):
    x = J[(J.task == task) & (J.model == f"delta: {name} minus portable")].iloc[0]; return x.auc, x.ci_lo, x.ci_hi


def orow(D, name):
    x = D[D["comparison"].str.startswith(name)].iloc[0]; return x.delta, x.ci_lo, x.ci_hi


comp = [("Amyloid: + compact AD-signature (5)", jrow("amyloid", "portable + compact (5)"), orow(dA, "portable + compact")),
        ("Amyloid: + full FreeSurfer panel", jrow("amyloid", "portable + full FreeSurfer (324)"), orow(dA, "portable + full FreeSurfer")),
        ("Trajectory: + compact AD-signature (5)", jrow("trajectory", "portable + compact (5)"), orow(dT, "portable + compact")),
        ("Trajectory: + full FreeSurfer panel", jrow("trajectory", "portable + full FreeSurfer (324)"), orow(dT, "portable + full FreeSurfer"))]
for i, (lab, a, o) in enumerate(comp):
    yy = len(comp) - i
    axC.errorbar(a[0], yy + 0.15, xerr=[[a[0] - a[1]], [a[2] - a[0]]], fmt="D", color="#7f7f7f", mfc="white", ms=5, capsize=2.5)
    axC.errorbar(o[0], yy - 0.15, xerr=[[o[0] - o[1]], [o[2] - o[0]]], fmt="o", color="#2c5496", ms=5, capsize=2.5)
axC.axvline(0, color="black", lw=0.8)
axC.set_yticks(range(len(comp), 0, -1)); axC.set_yticklabels([c[0] for c in comp], fontsize=7.8)
axC.set_xlabel("ΔAUC versus the portable model (refitted within each cohort; 95% CI)")
axC.plot([], [], "D", color="#7f7f7f", mfc="white", label="ADNI (n=1,059 / 612)")
axC.plot([], [], "o", color="#2c5496", label="OASIS-3 (n=96 / 117)")
axC.legend(loc="lower left", fontsize=7.6); axC.set_title("C  Incremental value of structural MRI", loc="left")
axC.spines[["top", "right"]].set_visible(False); axC.grid(axis="x", alpha=0.3)

for ext in ("png", "tif"):
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
    fig.savefig(os.path.join(FIG, f"Figure5.{ext}"), dpi=300, bbox_inches="tight", **kw)
print("Figure 5 saved")
