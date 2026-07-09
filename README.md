# Staging within mild cognitive impairment in ADNI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Reproducible, **subject-level, leakage-controlled** code for the study *"Staging within
Mild Cognitive Impairment from Routinely Available Data: Amyloid Status, Conversion
Trajectory, and the Limited Incremental Value of Structural MRI in ADNI."*

> **Central finding.** Clinically meaningful stages *within* MCI — amyloid positivity, a
> stable/slow/fast conversion trajectory, and a four-class CN–A−MCI–A+MCI–AD ordering — can
> be recovered from inexpensive, largely non-imaging features (chiefly APOE genotype and
> age; amyloid ROC-AUC ≈ 0.82). Standardized FreeSurfer morphometry, an end-to-end 3D CNN,
> and even the early rate of atrophy add **no incremental value** over these low-cost
> predictors, although amyloid-positive MCI do atrophy faster.

---

## ⚠️ Data availability and ethics (read first)

**This repository contains code only. It contains NO ADNI data and NO subject-level derived data.**

ADNI data are governed by the [ADNI Data Use Agreement](https://adni.loni.usc.edu/data-samples/access-data/).
To reproduce the analyses you must obtain the data yourself from
[adni.loni.usc.edu](https://adni.loni.usc.edu/) after approval:

- **Clinical / cognitive / genetic / imaging-derived tables** via the `ADNIMERGE2` R data
  package (read locally with `pyreadr`), plus CSF/PET/plasma biomarker tables.
- **Raw T1 MRI** (only for the optional imaging / CNN arm) via the ADNI image collections.

Subject-level tables, manifests, model out-of-fold predictions and trained weights are
**excluded by design** (see `.gitignore`) and must not be redistributed.

---

## Repository structure

```
code/            Tabular pipeline (Python) — the main analyses
  ml_common.py                 leakage-safe, subject-level CV engine (asserts train/test disjointness)
  01_extract_data.py           read ADNIMERGE2 .rda tables
  02_build_manifest_cohorts.py subject-level cohorts (one row per subject)
  03_build_features.py         feature families (demographics/APOE, cognition, FreeSurfer)
  19_extract_biomarkers.py     baseline CSF/PET/plasma biomarker table
  23_within_mci_staging.py     amyloid A+/A−, conversion trajectory, four-class ordering
  28_fs_change_staging.py      leakage-safe landmark FreeSurfer-change arm
  29_amyloid_utility.py        calibration, decision curve, operating point, APOE-stratified, ΔAUC, cohort table
  make_figures.py              Figure 1 (feature-family AUCs) and Figure 2 (clinical utility)
gpu_deep/        Optional imaging arm (Python + PyTorch/MONAI, GPU) — Supplementary Table S3
  build_deep_manifest.py, build_staging_manifest.py, staging_deep.py, RUN_4_staging.bat
run_local/       Optional CAT12 segmentation of baseline T1 scans (MATLAB + SPM12/CAT12)
  step1_cat12_segment.m, RUN_1_cat12.bat
requirements.txt Python dependencies
```

The tabular pipeline reproduces the main text (Tables 1–2, Figures 1–2) and Supplementary
Tables S1, S1b, S2, S4, S5, S6. The optional imaging arm reproduces Supplementary Table S3
(end-to-end 3D CNN staging) and requires raw T1 scans and an NVIDIA GPU.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+. The imaging arm additionally requires MATLAB with SPM12 + CAT12 (segmentation)
and a CUDA-enabled PyTorch build for the CNN.

## Reproducing the analyses (tabular; no raw images needed)

Run from the repository root (scripts expect `./data` and write to `./outputs`):

```bash
python code/01_extract_data.py            # read ADNIMERGE2 .rda tables  ->  data/
python code/02_build_manifest_cohorts.py  # subject-level cohorts (one row/subject)
python code/03_build_features.py          # feature families
python code/19_extract_biomarkers.py      # baseline biomarker table
python code/23_within_mci_staging.py      # amyloid / trajectory / four-class staging  -> outputs/within_mci/
python code/28_fs_change_staging.py       # leakage-safe landmark FreeSurfer-change arm
python code/29_amyloid_utility.py         # calibration, DCA, operating point, APOE-stratified, ΔAUC, Table 1
python code/make_figures.py               # Figure 1 and Figure 2  -> figures/
```

**Optional imaging arm (Supplementary Table S3; requires raw T1 + GPU):** segment baseline
scans with `run_local/RUN_1_cat12.bat`, then run `gpu_deep/RUN_4_staging.bat`.

## Leakage controls (design summary)

- **Subject-level partitioning** — one row per subject; no participant in both train and test
  (`code/ml_common.py` asserts train/test subject disjointness).
- **No circularity** — molecular biomarkers are never used as predictors of amyloid status.
- **Pre-specified primary classifier** — L2 logistic regression, to avoid model-selection
  optimism; two additional classifiers are reported in the supplement.
- **Leakage-safe landmark change** — early atrophy over `[0, L]` predicts conversion after `L`.
- **Fold-aligned fusion** (imaging arm) — deep embeddings are out-of-fold; fusion is fit on
  outer-train only, so each subject is predicted exactly once.

## Citation

> Dündar MS. *Staging within Mild Cognitive Impairment from Routinely Available Data: Amyloid
> Status, Conversion Trajectory, and the Limited Incremental Value of Structural MRI in ADNI.*
> (2026). Manuscript under review.

*(Will be updated with DOI/journal once available.)*

## License

Code is released under the [MIT License](LICENSE). This license covers **the code only**;
ADNI data remain governed by the ADNI Data Use Agreement and are not redistributed here.

## Author

**Mehmet Sait Dündar**, Erciyes University, Halil Bayraktar Health Services Vocational School,
Department of Medical Imaging Techniques, Kayseri, Türkiye.
ORCID: [0000-0002-0336-4825](https://orcid.org/0000-0002-0336-4825).

## Acknowledgement

Data used in preparation of this work were obtained from the Alzheimer's Disease Neuroimaging
Initiative (ADNI) database (adni.loni.usc.edu). The ADNI investigators contributed to the
design and implementation of ADNI and/or provided data but did not participate in the analysis
or writing of this work. A complete listing of ADNI investigators is available at the ADNI website.
