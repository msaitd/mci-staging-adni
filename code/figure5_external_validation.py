"""figure5_external_validation.py -- Figure 5 (external validation in OASIS-3 and NACC), drawn only from aggregate results:
OASIS-3 and NACC aggregate outputs (external_validation/oasis_results and nacc_results, produced by the two validation scripts), the frozen-model
reference AUCs (frozen_adni_models.json) and the ADNI portable-imaging comparison (J_portable_imaging.csv)."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EV = os.path.join(ROOT, "external_validation")
RES, NRES = os.path.join(EV, "oasis_results"), os.path.join(EV, "nacc_results")
FIG = os.path.join(ROOT, "figures", "revision")
plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold", "axes.titlesize": 9.5, "font.family": "DejaVu Sans"})

r = pd.read_csv(os.path.join(RES, "oasis_results.csv"))
q = pd.read_csv(os.path.join(NRES, "nacc_results.csv"))
fz = json.load(open(os.path.join(EV, "frozen_adni_models.json")))["models"]
J = pd.read_csv(os.path.join(ROOT, "outputs", "revision", "J_portable_imaging.csv"))
dec = pd.read_csv(os.path.join(RES, "oasis_calibration_deciles.csv"))
cal = pd.read_csv(os.path.join(RES, "oasis_calibration_summary.csv")).iloc[0]
ndec = pd.read_csv(os.path.join(NRES, "nacc_calibration_deciles.csv"))
ncal = pd.read_csv(os.path.join(NRES, "nacc_calibration_summary.csv")).iloc[0]
dA = pd.read_csv(os.path.join(RES, "oasis_fs_deltas_amyloid.csv"))
dT = pd.read_csv(os.path.join(RES, "oasis_fs_deltas_trajectory.csv"))
nA = pd.read_csv(os.path.join(NRES, "nacc_fs_deltas_amyloid.csv"))
nT = pd.read_csv(os.path.join(NRES, "nacc_fs_deltas_trajectory.csv"))

C_ADNI, C_OAS, C_NACC = "#7f7f7f", "#2c5496", "#c55a11"


def row(D, analysis_start, subset_start, model="portable"):
    k = D["analysis"].str.startswith(analysis_start) & D["subset"].str.startswith(subset_start) & (D["model"] == model)
    x = D[k].iloc[0]; return float(x.auc), float(x.ci_lo), float(x.ci_hi), int(x.n)


A1, T1, F1, N5 = "Aim1 amyloid (frozen", "Aim3 trajectory (frozen", "Aim4 four-group (frozen", "Aim5 autopsy (frozen"
O, N = "O", "N"
items = [("Amyloid status in MCI", None),
         ("ADNI, cross-validated (n=1,059)", ("ADNI", fz["amyloid"]["adni_cv_auc"], *fz["amyloid"]["adni_cv_auc_ci"])),
         ("OASIS-3 primary: PiB, CDR 0.5, MMSE ≥24", (O, A1, "primary cohort, Centiloid >= 20")),
         ("Centiloid ≥12", (O, A1, "primary cohort, Centiloid >= 12")),
         ("Centiloid ≥30", (O, A1, "primary cohort, Centiloid >= 30")),
         ("Partial-volume-corrected Centiloid", (O, A1, "primary cohort, PVC")),
         ("PET quality control passed", (O, A1, "PUP QC passed")),
         ("Clinical visit ≤90 days from PET", (O, A1, "clinical visit within 90")),
         ("Non-AD clinical aetiology excluded", (O, A1, "sensitivity: CDR 0.5, MMSE>=24, clinician non-AD")),
         ("Clinician label 'not demented'", (O, A1, "sensitivity: CDR 0.5, MMSE>=24, clinician label")),
         ("No MMSE criterion", (O, A1, "sensitivity: CDR 0.5, any MMSE")),
         ("Florbetapir PET", (O, A1, "AV45 only")),
         ("PiB and florbetapir pooled", (O, A1, "all tracers")),
         ("NACC primary: SCAN PET, CDR 0.5, MMSE ≥24", (N, A1, "primary cohort, Centiloid >= 20")),
         ("SCAN-provided amyloid status", (N, A1, "SCAN-provided")),
         ("Centiloid ≥12", (N, A1, "primary cohort, Centiloid >= 12")),
         ("Centiloid ≥30", (N, A1, "primary cohort, Centiloid >= 30")),
         ("PiB PET", (N, A1, "tracer code 2")),
         ("Florbetaben PET", (N, A1, "tracer code 4")),
         ("Florbetapir PET", (N, A1, "tracer code 3")),
         ("Clinical visit ≤90 days from PET", (N, A1, "clinical visit within 90")),
         ("Clinician MCI diagnosis", (N, A1, "sensitivity: clinician MCI")),
         ("No MMSE criterion", (N, A1, "sensitivity: CDR 0.5, not demented, any MMSE")),
         ("MoCA→MMSE crosswalk, lower 95% CI bound", (N, A1, "MoCA-derived MMSE at the lower")),
         ("MoCA→MMSE crosswalk, upper 95% CI bound", (N, A1, "MoCA-derived MMSE at the upper")),
         ("MoCA-derived MMSE not used", (N, A1, "MoCA-derived MMSE not used")),
         ("Mixed-protocol PET", (N, A1, "mixed-protocol PET only")),
         ("Standardized and mixed-protocol PET pooled", (N, A1, "standardized + mixed-protocol PET (image QC passed)")),
         ("Neuropathology at autopsy (NACC, last MCI visit)", None),
         ("Intermediate/high ADNC", (N, N5, "last MCI visit; reference: intermediate/high")),
         ("Intermediate/high ADNC, visit ≤5 years before death", (N, N5, "last MCI visit <=5 years before death; reference: intermediate/high")),
         ("Moderate/frequent neuritic plaques", (N, N5, "last MCI visit; reference: moderate/frequent")),
         ("Neuritic plaques, visit ≤5 years before death", (N, N5, "last MCI visit <=5 years before death; reference: moderate/frequent")),
         ("Stable/slow/fast trajectory", None),
         ("ADNI, cross-validated (n=612)", ("ADNI", fz["trajectory"]["adni_cv_auc"], *fz["trajectory"]["adni_cv_auc_ci"])),
         ("OASIS-3 primary: progression to CDR ≥1", (O, T1, "primary")),
         ("Clinician-diagnosed dementia also counted", (O, T1, "sensitivity")),
         ("NACC primary: clinician-diagnosed dementia", (N, T1, "primary")),
         ("Progression to CDR ≥1", (N, T1, "sensitivity: event = CDR global")),
         ("Clinician MCI diagnosis at baseline", (N, T1, "sensitivity: clinician MCI baseline")),
         ("Four groups (CN / A−MCI / A+MCI / AD)", None),
         ("ADNI, cross-validated (n=3,221)", ("ADNI", fz["fourgroup"]["adni_cv_auc"], *fz["fourgroup"]["adni_cv_auc_ci"])),
         ("OASIS-3", (O, F1, "CN")),
         ("NACC", (N, F1, "CN"))]

fig = plt.figure(figsize=(8.5, 11.6))   # portrait: fits one journal page at 6.5 in width
ax = fig.add_axes([0.42, 0.355, 0.45, 0.625]); axB = fig.add_axes([0.08, 0.035, 0.30, 0.225]); axC = fig.add_axes([0.66, 0.035, 0.32, 0.225])

y = len(items); ticks, labels = [], []
for lab, spec in items:
    if spec is None:
        ax.text(-0.012, y, lab, fontweight="bold", va="center", ha="right", fontsize=7.8, transform=ax.get_yaxis_transform())
    else:
        if spec[0] == "ADNI":
            v, lo, hi = spec[1], spec[2], spec[3]; mk, mfc, col = "D", "white", C_ADNI
        else:
            D = r if spec[0] == O else q
            v, lo, hi, n = row(D, spec[1], spec[2])
            mk, col = ("o", C_OAS) if spec[0] == O else ("s", C_NACC); mfc = col
            if "n=" not in lab:
                lab = f"{lab} (n={n:,})"
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt=mk, color=col, mfc=mfc, ms=3.8, capsize=1.8, lw=0.9)
        ax.text(1.012, y, f"{v:.2f} ({lo:.2f}–{hi:.2f})", va="center", fontsize=6.9)
        ticks.append(y); labels.append(lab)
    y -= 1
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.1)
ax.set_xlim(0.5, 1.13); ax.set_xticks(np.arange(0.5, 1.01, 0.1)); ax.set_ylim(0.3, len(items) + 0.7)
ax.axvline(0.5, color="grey", lw=0.6, ls="--")
ax.set_xlabel("ROC AUC (95% bootstrap CI); macro one-vs-rest for multiclass", fontsize=8)
ax.set_title("A  Frozen ADNI portable models in OASIS-3 and NACC", loc="left", fontsize=8.5, x=-0.9)
ax.plot([], [], "D", color=C_ADNI, mfc="white", label="ADNI (cross-validated reference)")
ax.plot([], [], "o", color=C_OAS, label="OASIS-3 (external, unchanged)")
ax.plot([], [], "s", color=C_NACC, label="NACC (external, unchanged)")
ax.legend(loc="upper center", bbox_to_anchor=(0.30, -0.042), ncol=3, fontsize=7.2, frameon=False, handletextpad=0.3, columnspacing=1.0)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=0.3)

axB.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8, label="Ideal")
axB.plot(dec["mean_pred"], dec["observed"], "o-", color=C_OAS, ms=3.8, lw=1, label="OASIS-3 (PiB, n=98)")
axB.plot(ndec["mean_pred"], ndec["observed"], "s-", color=C_NACC, ms=3.8, lw=1, label="NACC (SCAN PET, n=644)")
axB.set_xlim(0, 1); axB.set_ylim(0, 1.02); axB.set_xlabel("Predicted probability (A+), deciles", fontsize=8); axB.set_ylabel("Observed frequency", fontsize=8)
axB.tick_params(labelsize=7)
axB.text(0.03, 0.97, f"OASIS-3: Brier {cal.brier:.3f}, slope {cal.cal_slope:.2f}, CITL {cal.cal_intercept_citl:.2f}\n"
                     f"NACC:     Brier {ncal.brier:.3f}, slope {ncal.cal_slope:.2f}, CITL {ncal.cal_intercept_citl:.2f}",
         va="top", fontsize=6.6, bbox=dict(boxstyle="round", fc="white", ec="grey"))
axB.legend(loc="lower right", fontsize=6.6); axB.set_title("B  Calibration of the frozen amyloid model", loc="left", fontsize=8.5, x=-0.12)
axB.spines[["top", "right"]].set_visible(False)


def jrow(task, name):
    x = J[(J.task == task) & (J.model == f"delta: {name} minus portable")].iloc[0]; return x.auc, x.ci_lo, x.ci_hi


def orow(D, name):
    x = D[D["comparison"].str.startswith(name)].iloc[0]; return x.delta, x.ci_lo, x.ci_hi


comp = [("Amyloid: + compact AD-signature (5)", jrow("amyloid", "portable + compact (5)"), orow(dA, "portable + compact"), orow(nA, "portable + compact")),
        ("Amyloid: + full FreeSurfer panel", jrow("amyloid", "portable + full FreeSurfer (324)"), orow(dA, "portable + full FreeSurfer"), orow(nA, "portable + full FreeSurfer")),
        ("Trajectory: + compact AD-signature (5)", jrow("trajectory", "portable + compact (5)"), orow(dT, "portable + compact"), orow(nT, "portable + compact")),
        ("Trajectory: + full FreeSurfer panel", jrow("trajectory", "portable + full FreeSurfer (324)"), orow(dT, "portable + full FreeSurfer"), orow(nT, "portable + full FreeSurfer"))]
for i, (lab, a, o, nn) in enumerate(comp):
    yy = len(comp) - i
    for (e, lo, hi), off, mk, col, mfc in ((a, 0.22, "D", C_ADNI, "white"), (o, 0.0, "o", C_OAS, C_OAS), (nn, -0.22, "s", C_NACC, C_NACC)):
        axC.errorbar(e, yy + off, xerr=[[e - lo], [hi - e]], fmt=mk, color=col, mfc=mfc, ms=3.8, capsize=1.8, lw=0.9)
axC.axvline(0, color="black", lw=0.8)
axC.set_yticks(range(len(comp), 0, -1)); axC.set_yticklabels([c[0] for c in comp], fontsize=7); axC.tick_params(axis="x", labelsize=7)
axC.set_xlabel("ΔAUC vs portable model, refitted in each cohort (95% CI)", fontsize=7.4)
axC.plot([], [], "D", color=C_ADNI, mfc="white", label="ADNI (n=1,059 / 612)")
axC.plot([], [], "o", color=C_OAS, label="OASIS-3 (n=96 / 117)")
axC.plot([], [], "s", color=C_NACC, label="NACC (n=335 / 103)")
axC.legend(loc="upper left", fontsize=6.4, framealpha=0.95); axC.set_title("C  Incremental value of structural MRI", loc="left", fontsize=8.5, x=-0.6)
axC.spines[["top", "right"]].set_visible(False); axC.grid(axis="x", alpha=0.3)

for ext in ("png", "tif"):
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
    fig.savefig(os.path.join(FIG, f"Figure5.{ext}"), dpi=300, bbox_inches="tight", **kw)
print("Figure 5 saved")
