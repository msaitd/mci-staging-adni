"""revision_figures.py -- figures of the revised manuscript (no titles inside images; captions
are given in the manuscript). Uses only cached out-of-fold / held-out results."""
from revision_common import *
from revision_common import _bin_auc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
import atrophy_defs as E

FIG = os.path.join(ROOT, "figures", "revision"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "font.family": "DejaVu Sans"})
A = pd.read_csv(os.path.join(OUT, "A_primary_repeated_apparent.csv"))
Em = pd.read_csv(os.path.join(OUT, "E_atrophy_models.csv"))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".tif"), dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# ---------------- Figure 1: feature-family comparison ----------------
LAB = {"demo_apoe": "Demo-\ngraphics\n+ APOE", "cognition": "Cognition", "freesurfer": "Free-\nSurfer\n(324)",
       "clinical": "Clinical", "clinical+fs": "Clinical\n+ Free-\nSurfer (324)", "all": "Clinical\n+ FS\n+ bio-\nmarkers",
       "clinical+compact_raw": "Clinical\n+ AD-\nsignature (5)"}
COL = {"demo_apoe": "#8da0b3", "cognition": "#4f7cac", "freesurfer": "#b0b0b0", "clinical": "#2c5496",
       "clinical+fs": "#6a9f58", "all": "#c07038", "clinical+compact_raw": "#8e5ea2"}
panels = [("amyloid", "A  Amyloid status in MCI (A+ vs A−; n = 1,059)", ["demo_apoe", "cognition", "freesurfer", "clinical", "clinical+fs", "clinical+compact_raw"]),
          ("trajectory", "B  Conversion trajectory (stable/slow/fast; n = 612)", ["demo_apoe", "cognition", "freesurfer", "clinical", "clinical+fs", "all", "clinical+compact_raw"]),
          ("fourclass", "C  CN / A−MCI / A+MCI / AD (n = 3,221)", ["demo_apoe", "cognition", "freesurfer", "clinical", "clinical+fs", "clinical+compact_raw"])]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
for ax, (task, title, sets) in zip(axes, panels):
    vals = []
    for s in sets:
        r = (Em[(Em.task == task) & (Em.featureset == s)] if s == "clinical+compact_raw" else A[(A.task == task) & (A.featureset == s)]).iloc[0]
        v = r["auc"] if s == "clinical+compact_raw" else r["oof_auc"]
        vals.append((v, r["ci_lo"], r["ci_hi"]))
    x = np.arange(len(sets))
    for i, (s, (v, lo, hi)) in enumerate(zip(sets, vals)):
        ax.bar(i, v, color=COL[s], edgecolor="black", linewidth=0.6, width=0.72)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color="black", capsize=3, lw=1)
        ax.text(i, hi + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([LAB[s] for s in sets], rotation=0, fontsize=7.8)
    ax.axhline(0.5, ls="--", color="grey", lw=0.8); ax.set_ylim(0.4, 1.04); ax.set_title(title, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Cross-validated ROC AUC\n(out-of-fold predictions; macro one-vs-rest for multiclass)")
fig.tight_layout(); save(fig, "Figure1")

# ---------------- Figure 3: clinical utility of the amyloid model ----------------
d = pd.read_csv(os.path.join(OUT, "oof_amyloid_clinical.csv")); y = (d.y_true == "A+").astype(int).values; p = d["p_A+"].values
fpr, tpr, thr = roc_curve(y, p); a = _bin_auc(y == 1, p)
t90 = max(t for t in np.unique(p) if ((p >= t) & (y == 1)).sum() / y.sum() >= 0.90)
sens = ((p >= t90) & (y == 1)).sum() / y.sum(); spec = ((p < t90) & (y == 0)).sum() / (y == 0).sum()
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4))
ax[0].plot(fpr, tpr, color="#2c5496", lw=2, label=f"Clinical model (AUC {a:.2f})"); ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
ax[0].scatter([1 - spec], [sens], color="#c0392b", zorder=5)
ax[0].annotate(f"90% sensitivity\n(specificity {spec:.2f})", (1 - spec, sens), (0.58, 0.62), arrowprops=dict(arrowstyle="->", color="#c0392b"), fontsize=8)
ax[0].set_xlabel("1 − specificity"); ax[0].set_ylabel("Sensitivity"); ax[0].legend(loc="lower right", fontsize=8); ax[0].set_title("A  Discrimination (ROC)", loc="left")
dec = pd.qcut(p, 10, duplicates="drop"); rel = pd.DataFrame({"p": p, "y": y, "b": dec}).groupby("b", observed=True).agg(pp=("p", "mean"), oo=("y", "mean"))
lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
lr = LogisticRegression().fit(lg, y); br = np.mean((p - y) ** 2)
ax[1].plot([0, 1], [0, 1], "--", color="grey", lw=0.8, label="Ideal"); ax[1].plot(rel.pp, rel.oo, "o-", color="#2c5496", label="Observed (deciles)")
ax[1].text(0.04, 0.83, f"Brier {br:.3f}\nslope {lr.coef_[0, 0]:.2f}\nintercept {lr.intercept_[0]:.2f}", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="grey"))
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1); ax[1].set_xlabel("Predicted probability (A+)"); ax[1].set_ylabel("Observed frequency")
ax[1].legend(loc="lower right", fontsize=8); ax[1].set_title("B  Calibration", loc="left")
ths = np.linspace(0.01, 0.60, 60); prev = y.mean(); N = len(y)
nb = [((p >= t) & (y == 1)).sum() / N - ((p >= t) & (y == 0)).sum() / N * t / (1 - t) for t in ths]
na = [prev - (1 - prev) * t / (1 - t) for t in ths]
ax[2].plot(ths, nb, color="#2c5496", lw=2, label="Clinical model"); ax[2].plot(ths, na, "--", color="#6a9f58", lw=1.5, label="Test all")
ax[2].axhline(0, ls=":", color="grey", lw=1, label="Test none"); ax[2].set_ylim(-0.05, 0.62); ax[2].set_xlim(0, 0.6)
ax[2].set_xlabel("Threshold probability"); ax[2].set_ylabel("Net benefit"); ax[2].legend(fontsize=8); ax[2].set_title("C  Decision curve", loc="left")
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); save(fig, "Figure3")

# ---------------- Figure 2: ROC curves for all stagings ----------------
X, fam = load_all(); X = E.add_compact(X); demo, cog = fam["demo"], fam["cognition"]
M = X[X["baseline_dx"].eq("MCI") & X["A"].notna()].reset_index(drop=True); yM = np.where(M.A == 1, "A+", "A-")
Pcomp, _ = oof_predict(M[demo + cog + E.COMPACT], yM, repeated_kfold_first(yM), "logistic_l2", ["A-", "A+"], M.RID.values)
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4))
curves = [("clinical", "Clinical", "#2c5496", None), ("freesurfer", "FreeSurfer (324)", "#8a8a8a", None),
          ("clinical+fs", "Clinical + FreeSurfer (324)", "#6a9f58", None), ("compact", "Clinical + AD-signature (5)", "#8e5ea2", Pcomp)]
for key, lab, col, Pc in curves:
    if Pc is None:
        dd = pd.read_csv(os.path.join(OUT, f"oof_amyloid_{key}.csv")); yy = (dd.y_true == "A+").values; pp = dd["p_A+"].values
    else:
        yy = yM == "A+"; pp = Pc[:, 1]
    f_, t_, _ = roc_curve(yy, pp); ax[0].plot(f_, t_, color=col, lw=1.8, label=f"{lab}: {_bin_auc(yy, pp):.2f}")
ax[0].set_title("A  Amyloid status in MCI", loc="left")
for i, (task, classes, title) in enumerate([("trajectory", ["stable", "slow", "fast"], "B  Conversion trajectory (clinical model)"),
                                           ("fourclass", ["CN", "A-MCI", "A+MCI", "AD"], "C  CN / A−MCI / A+MCI / AD (clinical model)")], start=1):
    dd = pd.read_csv(os.path.join(OUT, f"oof_{task}_clinical.csv")); yy = dd.y_true.astype(str).values
    cols_ = ["#2c5496", "#e0a800", "#c0392b", "#6a9f58"]
    for k, c in enumerate(classes):
        f_, t_, _ = roc_curve(yy == c, dd["p_" + c]); lab = c.replace("-", "−")
        ax[i].plot(f_, t_, color=cols_[k], lw=1.8, label=f"{lab} vs rest: {_bin_auc(yy == c, dd['p_' + c].values):.2f}")
    mac = auc(yy, dd[["p_" + c for c in classes]].values, classes)
    ax[i].set_title(title, loc="left"); ax[i].text(0.40, 0.08, f"macro one-vs-rest AUC {mac:.2f}", fontsize=8)
for a_ in ax:
    a_.plot([0, 1], [0, 1], "--", color="grey", lw=0.8); a_.set_xlabel("1 − specificity"); a_.set_ylabel("Sensitivity")
    a_.legend(loc="lower right", fontsize=7.5, title="ROC AUC", title_fontsize=7.5, bbox_to_anchor=(1.0, 0.14))
    a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); save(fig, "Figure2")

# ---------------- Figure 4: internal-external validation and amyloid cut-offs ----------------
B = pd.read_csv(os.path.join(OUT, "B_internal_external.csv")); Cc = pd.read_csv(os.path.join(OUT, "C_cutoff_sensitivity.csv"))
fig, ax = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={"width_ratios": [1, 1.25]})
rows = []
for task, tl in (("amyloid", "Amyloid"), ("trajectory", "Trajectory"), ("fourclass", "Four-group")):
    for design, dl in (("5fold", "stratified 5-fold"), ("LOSO", "leave-one-site-out"), ("leave-phase-out", "leave-one-phase-out"), ("temporal", "temporal (ADNI-1/GO/2→3/4)")):
        for fsn in ("clinical", "clinical+fs"):
            if design == "5fold":
                r = A[(A.task == task) & (A.featureset == fsn)].iloc[0]; v, lo, hi = r.oof_auc, r.ci_lo, r.ci_hi
            else:
                r = B[(B.task == task) & (B.design == design) & (B.featureset == fsn)].iloc[0]; v, lo, hi = r.auc, r.ci_lo, r.ci_hi
            rows.append((f"{tl}: {dl}", fsn, v, lo, hi))
labels = list(dict.fromkeys(r[0] for r in rows)); ypos = {l: len(labels) - i for i, l in enumerate(labels)}
for lab, fsn, v, lo, hi in rows:
    yy = ypos[lab] + (0.15 if fsn == "clinical" else -0.15)
    ax[0].errorbar(v, yy, xerr=[[v - lo], [hi - v]], fmt="o" if fsn == "clinical" else "s", color="#2c5496" if fsn == "clinical" else "#6a9f58",
                   mfc="#2c5496" if fsn == "clinical" else "white", ms=5, capsize=2, lw=1)
ax[0].set_yticks(list(ypos.values())); ax[0].set_yticklabels(list(ypos.keys()), fontsize=8)
ax[0].set_xlabel("ROC AUC on held-out participants (95% bootstrap CI)"); ax[0].set_title("A  Internal–external validation", loc="left")
ax[0].plot([], [], "o", color="#2c5496", label="Clinical"); ax[0].plot([], [], "s", color="#6a9f58", mfc="white", label="Clinical + FreeSurfer (324)")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=False); ax[0].set_xlim(0.55, 1.0)
short = {r: r.replace("12-30", "12–30").replace("PET-CSF", "PET–CSF").replace("Abeta42", "Aβ42").replace(">=", "≥").replace("+/-", "±").replace("p-tau181", "p-tau181") for r in Cc.definition}
for i, r in enumerate(Cc.itertuples()):
    yy = len(Cc) - i
    ax[1].errorbar(r.auc_clinical, yy + 0.12, xerr=[[r.auc_clinical - r.ci_lo], [r.ci_hi - r.auc_clinical]], fmt="o", color="#2c5496", ms=5, capsize=2, lw=1)
    ax[1].plot(r.auc_clinical_fs, yy - 0.12, "s", color="#6a9f58", mfc="white", ms=5)
ax[1].set_yticks(range(len(Cc), 0, -1)); ax[1].set_yticklabels([f"{short[d]} (n={n})" for d, n in zip(Cc.definition, Cc.n)], fontsize=7.6)
ax[1].axvline(Cc.auc_clinical.iloc[0], color="#2c5496", ls=":", lw=0.8)
ax[1].set_xlabel("ROC AUC, amyloid status in MCI (clinical model with 95% CI)"); ax[1].set_title("B  Alternative amyloid definitions and cut-offs", loc="left")
ax[1].plot([], [], "o", color="#2c5496", label="Clinical"); ax[1].plot([], [], "s", color="#6a9f58", mfc="white", label="Clinical + FreeSurfer (324)")
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=False); ax[1].set_xlim(0.65, 0.92)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False); a_.grid(axis="x", alpha=0.3)
fig.tight_layout(); save(fig, "Figure4")
print("figures saved to", FIG)
