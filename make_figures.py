"""make_figures.py  —  Manuscript B figures.

Figure 1: within-MCI staging by feature family (pre-specified logistic primary),
          from outputs/within_mci/within_mci_summary_v2.csv (produced by 23_within_mci_staging.py).
Figure 2: clinical utility of the amyloid-status model (ROC + calibration + decision curve),
          recomputed out-of-fold from the tabular data via the leakage-safe CV engine.

Layout convention (same as 23/28/29): this file lives in <repo>/code/, reads
<repo>/data/*.csv and <repo>/outputs/within_mci/*, writes <repo>/figures/*.
"""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from ml_common import evaluate_combo
from sklearn.metrics import roc_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data"); OUT = os.path.join(ROOT, "outputs", "within_mci")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- Figure 1
d = pd.read_csv(os.path.join(OUT, "within_mci_summary_v2.csv"))
d = d[d.model == "logistic_l2"].copy()
panels = [("amyloid_mci", "Amyloid A+/A- MCI\n(n=1,059)"),
          ("trajectory", "Conversion trajectory\nstable/slow/fast (n=612)"),
          ("ordinal", "Four-class CN-A-MCI-A+MCI-AD\n(n=3,221)")]
famorder = ["demo_apoe", "cognition", "freesurfer", "clinical", "clinical+fs", "all"]
famlab = {"demo_apoe": "Demographics+APOE", "cognition": "Cognition", "freesurfer": "FreeSurfer",
          "clinical": "Clinical", "clinical+fs": "Clinical+FreeSurfer", "all": "Clinical+FS+biomarker"}
colors = {"demo_apoe": "#8c9eb2", "cognition": "#4c78a8", "freesurfer": "#b0b0b0",
          "clinical": "#2f5597", "clinical+fs": "#6a9f58", "all": "#c0703a"}
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
for ax, (task, title) in zip(axes, panels):
    sub = d[d.task == task]; fams = [f for f in famorder if f in set(sub.featureset)]
    xs = np.arange(len(fams))
    a = [float(sub[sub.featureset == f].AUC.iloc[0]) for f in fams]
    lo = [float(sub[sub.featureset == f].AUC_lo.iloc[0]) for f in fams]
    hi = [float(sub[sub.featureset == f].AUC_hi.iloc[0]) for f in fams]
    err = [np.array(a) - np.array(lo), np.array(hi) - np.array(a)]
    ax.bar(xs, a, yerr=err, capsize=3, color=[colors[f] for f in fams], edgecolor="black", linewidth=0.6)
    for x, v in zip(xs, a): ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, ls="--", lw=0.8, color="0.5")
    ax.set_xticks(xs); ax.set_xticklabels([famlab[f] for f in fams], rotation=40, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10); ax.set_ylim(0.4, 1.02)
axes[0].set_ylabel("Out-of-fold ROC-AUC (macro OVR for multiclass)", fontsize=10)
fig.suptitle("Within-MCI staging by feature family (pre-specified logistic-regression primary model)", fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_within_mci.png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(FIG, "fig_within_mci.pdf"), bbox_inches="tight")
print("wrote figures/fig_within_mci.{png,pdf}")

# ---------------------------------------------------------------- Figure 2
m = pd.read_csv(os.path.join(D, "master_features.csv"), low_memory=False)
b = pd.read_csv(os.path.join(D, "biomarkers_baseline.csv"))
fam = json.load(open(os.path.join(D, "feature_families.json")))
demo, cog = fam["demo"], fam["cognition"]
X = m.merge(b, on="RID", how="left")
def amy(r):
    if pd.notna(r.get("AMYPET_CENTILOID")): return 1.0 if r["AMYPET_CENTILOID"] >= 20 else 0.0
    if pd.notna(r.get("CSF_ABETA42")): return 1.0 if r["CSF_ABETA42"] < 980 else 0.0
    return np.nan
X["A"] = X.apply(amy, axis=1)
sub = X[X.baseline_dx.eq("MCI") & X.A.notna()].copy()
y = np.where(sub["A"].values == 1.0, "A+", "A-"); g = sub["RID"].values
Xd = sub[demo + cog].apply(pd.to_numeric, errors="coerce"); Xd = Xd.fillna(Xd.median(numeric_only=True))
_, oof = evaluate_combo(Xd, y, g, ["A-", "A+"], "logistic_l2")
yt = (oof.y_true == "A+").astype(int).values; p = oof["p_A+"].values
N = len(yt); prev = yt.mean(); auc = roc_auc_score(yt, p)
fpr, tpr, _ = roc_curve(yt, p)
op = None
for t in sorted(np.unique(p)):
    yh = (p >= t).astype(int); tp = ((yh == 1) & (yt == 1)).sum(); fn = ((yh == 0) & (yt == 1)).sum()
    tn = ((yh == 0) & (yt == 0)).sum(); fp = ((yh == 1) & (yt == 0)).sum()
    if tp / (tp + fn) >= 0.90: op = (1 - tn / (tn + fp), tp / (tp + fn), tn / (tn + fp))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
eps = 1e-6; logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
lr = LogisticRegression().fit(logit.reshape(-1, 1), yt)
cal_slope, cal_int, brier = float(lr.coef_[0, 0]), float(lr.intercept_[0]), brier_score_loss(yt, p)
df = pd.DataFrame({"p": p, "y": yt}); df["bin"] = pd.qcut(df.p, 10, duplicates="drop")
rel = df.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean")).reset_index(drop=True)
ths = np.arange(0.01, 0.601, 0.01)
nb = [((p >= t).astype(int) @ yt) / N - (((p >= t).astype(int)) @ (1 - yt)) / N * (t / (1 - t)) for t in ths]
nb_all = [prev - (1 - prev) * (t / (1 - t)) for t in ths]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].plot(fpr, tpr, color="#2f5597", lw=2, label=f"Clinical model (AUC {auc:.2f})")
ax[0].plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
if op: ax[0].scatter([op[0]], [op[1]], color="#c0392b", zorder=5, s=45); ax[0].annotate(f"90% sensitivity\n(specificity {op[2]:.2f})", xy=(op[0], op[1]), xytext=(op[0] + 0.05, op[1] - 0.22), fontsize=8, arrowprops=dict(arrowstyle="->", color="#c0392b"))
ax[0].set_xlabel("1 - specificity"); ax[0].set_ylabel("Sensitivity"); ax[0].set_title("A  Discrimination (ROC)", fontsize=11, loc="left"); ax[0].legend(fontsize=8, loc="lower right")
ax[1].plot([0, 1], [0, 1], ls="--", color="0.6", lw=1, label="Ideal")
ax[1].plot(rel.pred, rel.obs, "o-", color="#2f5597", lw=1.8, ms=5, label="Observed (deciles)")
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1); ax[1].set_xlabel("Predicted probability (A+)"); ax[1].set_ylabel("Observed frequency")
ax[1].set_title("B  Calibration", fontsize=11, loc="left")
ax[1].text(0.05, 0.93, f"Brier {brier:.3f}\nslope {cal_slope:.2f}\nintercept {cal_int:.2f}", fontsize=8, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax[1].legend(fontsize=8, loc="lower right")
ax[2].plot(ths, nb, color="#2f5597", lw=2, label="Clinical model")
ax[2].plot(ths, nb_all, color="#6a9f58", lw=1.5, ls="--", label="Test all")
ax[2].axhline(0, color="0.6", lw=1, ls=":", label="Test none")
ax[2].set_ylim(-0.05, max(nb) + 0.05); ax[2].set_xlim(0, 0.6)
ax[2].set_xlabel("Threshold probability"); ax[2].set_ylabel("Net benefit"); ax[2].set_title("C  Decision curve", fontsize=11, loc="left"); ax[2].legend(fontsize=8)
fig.suptitle("Clinical utility of the amyloid-status model in MCI (out-of-fold, pre-specified logistic model)", fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_amyloid_utility.png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(FIG, "fig_amyloid_utility.pdf"), bbox_inches="tight")
print("wrote figures/fig_amyloid_utility.{png,pdf}")
