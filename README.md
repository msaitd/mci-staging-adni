# Staging within mild cognitive impairment in ADNI, OASIS-3 and NACC

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Reproducible, **subject-level, leakage-controlled** code for the study *"Staging within
Mild Cognitive Impairment from Routinely Available Data: Amyloid Status, Conversion
Trajectory, and the Limited Incremental Value of Structural MRI in ADNI, OASIS-3 and NACC."*

> **Central finding.** Clinically meaningful stages *within* MCI — amyloid positivity, a
> stable/slow/fast conversion trajectory, and a four-group CN / A−MCI / A+MCI / AD separation —
> can be recovered from inexpensive, largely non-imaging features (chiefly APOE genotype and
> age; amyloid ROC AUC ≈ 0.82, maintained under leave-one-site-out, leave-one-phase-out and
> temporal validation, across 16 amyloid definitions and against autopsy). A six-variable
> portable model developed in ADNI transferred unchanged to two external cohorts, OASIS-3
> (amyloid AUC 0.84 in PiB-PET MCI; trajectory 0.82) and NACC (amyloid AUC 0.77 in SCAN-PET MCI,
> n=644; trajectory 0.76, n=6,304), well calibrated in both. The full FreeSurfer
> panel, an end-to-end 3D CNN and the early rate of atrophy add no incremental value; a compact
> five-measure AD-signature atrophy summary adds a small prognostic increment in ADNI
> (0.84 → 0.86) that was not reproduced in OASIS-3 or NACC, and does not help identify amyloid status.

---

## ⚠️ Data availability and ethics (read first)

**This repository contains code only. It contains NO ADNI, OASIS or NACC data and NO subject-level derived data.**
The only data-derived file is `external_validation/frozen_adni_models.json`, which holds the fitted
parameters of the frozen ADNI portable models (coefficients, imputation medians, scaling constants).
`external_validation/moca_mmse_crosswalk.csv` reproduces the published MoCA-to-MMSE equipercentile table
(Monsell et al., *Alzheimer Dis Assoc Disord* 2016; 30: 134–139, Table 3).

ADNI data are governed by the [ADNI Data Use Agreement](https://adni.loni.usc.edu/data-samples/access-data/).
To reproduce the analyses you must obtain the data yourself from
[adni.loni.usc.edu](https://adni.loni.usc.edu/) after approval:

- **Clinical / cognitive / genetic / imaging-derived tables** via the `ADNIMERGE2` R data
  package (read locally with `pyreadr`), plus CSF/PET/plasma biomarker tables.
- **Raw T1 MRI** (only for the optional imaging / CNN arm) via the ADNI image collections.

Subject-level tables, manifests, model out-of-fold predictions and trained weights are
**excluded by design** (see `.gitignore`) and must not be redistributed.

OASIS-3 data (external validation) are governed by the [OASIS Data Use Agreement](https://sites.wustl.edu/oasisbrains/)
and are obtained from NITRC-IR after approval. Under that agreement only the authorized user may
access the data; `external_validation/oasis3_external_validation.py` is therefore run locally by
the data user and exports aggregate results only. Publications using florbetapir (AV45) data
must be provided to Avid Radiopharmaceuticals 30 days before publication.

NACC data (second external validation: Uniform Data Set, SCAN amyloid PET and MRI, neuropathology) are
obtained from [naccdata.org](https://naccdata.org/) after approval of a data request and are for non-commercial
use; manuscripts must be submitted to NACC before journal submission. `external_validation/nacc_external_validation.py`
is run locally by the data user and exports aggregate results only (counts < 5 masked).

---

## Repository structure

```
code/            Tabular pipeline (Python) — the main analyses
  ml_common.py                     leakage-safe, subject-level CV engine (asserts train/test disjointness)
  extract_data.py                  read ADNIMERGE2 .rda tables
  build_manifest_cohorts.py        subject-level cohorts (one row per subject)
  build_features.py                feature families (demographics/APOE, cognition, FreeSurfer)
  extract_biomarkers.py            baseline CSF/PET/plasma biomarker table
  within_mci_staging.py            amyloid A+/A−, conversion trajectory, four-group staging (3 classifiers)
  fs_change_staging.py             leakage-safe landmark FreeSurfer-change arm
  amyloid_utility.py               calibration, decision curve, operating point, APOE-stratified, ΔAUC, cohort table
  --- revision analyses ---
  revision_data.py                 amyloid PET/CSF detail (dates, tracer), neuropathology, diagnosis detail
  revision_common.py               shared loading and fold-internal helpers
  atrophy_defs.py                  compact AD-signature atrophy measures
  validation_repeated_apparent.py  out-of-fold vs per-fold, repeated-partition and apparent AUC (Table S8)
  validation_internal_external.py  leave-one-site-out, leave-one-phase-out, temporal validation (Table S9, Fig. 4A)
  amyloid_cutoff_sensitivity.py    16 PET/CSF amyloid definitions and cut-offs (Table S3, Fig. 4B)
  nonad_dementia_autopsy.py        amyloid-negative (non-AD) dementia; autopsy validation (Tables S10–S11)
  atrophy_redundancy.py            W-scores, compact atrophy, redundancy, age-adjusted atrophy rates (Tables S5b, S12)
  atrophy_compact_robustness.py    robustness of the compact-atrophy increment (Table S12b)
  fourgroup_compact_atrophy.py     four-group staging with compact atrophy (Table 2)
  age_matching_portable_model.py   age-matched analysis, age tertiles, portable model (Table S13)
  table_s1_bootstrap.py            Table S1 with 2,000-resample bootstrap CIs
  revision_figures.py              Figures 1–4
  --- external validation (OASIS-3, NACC) ---
  export_frozen_models.py          fit and freeze the ADNI portable models -> external_validation/frozen_adni_models.json
  portable_imaging_adni.py         ADNI reference: portable model + compact / full FreeSurfer (Table S15e)
  figure5_external_validation.py   Figure 5 from the aggregate OASIS-3 and NACC results
external_validation/
  oasis3_external_validation.py    run locally on the OASIS-3 tables; applies the frozen models (Table S15, Figure 5)
  nacc_external_validation.py      run locally on the NACC files; applies the frozen models (Table S16, Figure 5)
  moca_mmse_crosswalk.csv          published MoCA -> MMSE crosswalk used for NACC UDS v3+ visits
  frozen_adni_models.json          frozen ADNI portable-model parameters (no participant data)
gpu_deep/        Optional imaging arm (Python + PyTorch/MONAI, GPU) — Supplementary Table S4
  build_deep_manifest.py, build_staging_manifest.py, staging_deep.py
preproc/         Optional CAT12 segmentation of baseline T1 scans (MATLAB + SPM12/CAT12)
  cat12_segment.m
requirements.txt Python dependencies
```

The tabular pipeline reproduces the main text (Tables 1–2, Figures 1–5) and Supplementary Tables S1–S3 and S5–S16 (Figure 5 and Tables S15–S16 additionally require authorized OASIS-3 and NACC downloads). The optional imaging arm reproduces Supplementary Table S4 (end-to-end 3D CNN staging) and requires raw T1 scans and an NVIDIA GPU.

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
python code/extract_data.py                   # read ADNIMERGE2 .rda tables  ->  data/
python code/build_manifest_cohorts.py         # subject-level cohorts (one row/subject)
python code/build_features.py                 # feature families
python code/extract_biomarkers.py             # baseline biomarker table
python code/within_mci_staging.py             # amyloid / trajectory / four-group staging  -> outputs/within_mci/
python code/fs_change_staging.py              # leakage-safe landmark FreeSurfer-change arm
python code/amyloid_utility.py                # calibration, DCA, operating point, APOE-stratified, ΔAUC, Table 1
python code/revision_data.py                  # revision inputs (label timing/tracer, neuropathology)
python code/validation_repeated_apparent.py   # -> outputs/revision/  (also writes out-of-fold files used below)
python code/validation_internal_external.py   # resumable; re-run until it prints "B done"
python code/amyloid_cutoff_sensitivity.py
python code/nonad_dementia_autopsy.py
python code/atrophy_redundancy.py
python code/atrophy_compact_robustness.py
python code/fourgroup_compact_atrophy.py
python code/age_matching_portable_model.py
python code/table_s1_bootstrap.py
python code/revision_figures.py               # Figures 1–4 -> figures/revision/
python code/export_frozen_models.py           # frozen ADNI portable models -> external_validation/
python code/portable_imaging_adni.py          # ADNI reference for the imaging comparison
```

## External validation in OASIS-3 (run by the authorized OASIS data user)

```bash
# download the OASIS-3 tables (0AS_data_files) from NITRC-IR, then:
set OASIS_ROOT=path\to\OASIS3_data          # Windows  (Linux/macOS: export OASIS_ROOT=...)
python external_validation/oasis3_external_validation.py   # -> external_validation/oasis_results/ (aggregate only)
```

## External validation in NACC (run by the authorized NACC data user)

```bash
# download the NACC Quick-Access files (UDS investigator file, SCAN amyloid PET GAAIN and MRI FreeSurfer files), then:
set NACC_ROOT=path\to\NACC_data            # Windows  (Linux/macOS: export NACC_ROOT=...)
python external_validation/nacc_external_validation.py     # -> external_validation/nacc_results/ (aggregate only)
python code/figure5_external_validation.py                 # Figure 5 (needs both oasis_results and nacc_results)
```

**Optional imaging arm (Supplementary Table S3; requires raw T1 + GPU):** segment baseline
scans with `preproc/cat12_segment.m`, then run `gpu_deep/build_staging_manifest.py` followed by `gpu_deep/staging_deep.py`.

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

> Dündar MS, Yılmaz B. *Staging within Mild Cognitive Impairment from Routinely Available Data:
> Amyloid Status, Conversion Trajectory, and the Limited Incremental Value of Structural MRI in
> ADNI, OASIS-3 and NACC.* (2026). Manuscript under review (revised).

*(Will be updated with DOI/journal once available.)*

## License

Code is released under the [MIT License](LICENSE). This license covers **the code only**;
ADNI, OASIS and NACC data remain governed by their Data Use Agreements and are not redistributed here.

## Authors

**Mehmet Sait Dündar¹**, **Bülent Yılmaz²**

1. Erciyes University, Halil Bayraktar Health Services Vocational School, Department of Medical Imaging Techniques, Kayseri, Türkiye. ORCID: [0000-0002-0336-4825](https://orcid.org/0000-0002-0336-4825).
2. Department of Electrical and Computer Engineering, Gulf University for Science and Technology (GUST), Hawally, Kuwait.

## Acknowledgement

Data used in preparation of this work were obtained from the Alzheimer's Disease Neuroimaging
Initiative (ADNI) database (adni.loni.usc.edu). The ADNI investigators contributed to the
design and implementation of ADNI and/or provided data but did not participate in the analysis
or writing of this work. A complete listing of ADNI investigators is available at the ADNI website.

External validation data were provided in part by OASIS-3: Longitudinal Multimodal Neuroimaging: Principal
Investigators: T. Benzinger, D. Marcus, J. Morris; NIH P30 AG066444, P50 AG00561, P30 NS09857781, P01 AG026276,
P01 AG003991, R01 AG043434, UL1 TR000448, R01 EB009352. AV-45 doses were provided by Avid Radiopharmaceuticals,
a wholly owned subsidiary of Eli Lilly.

The NACC database is funded by NIA/NIH Grant U24 AG072122. NACC data are contributed by the NIA-funded ADRCs;
SCAN is a multi-institutional project that was funded as a U24 grant (AG067418) by the National Institute on Aging in May 2020.
The complete NACC, SCAN and CLARiTI acknowledgment statements are given in the Acknowledgments of the article.
