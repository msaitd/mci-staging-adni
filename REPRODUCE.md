# Reproduction notes

All results in the manuscript are reproduced from an authorized ADNI download. No data are
included in this repository.

## 1. Obtain the data (ADNI Data Use Agreement)

1. Apply for access at https://adni.loni.usc.edu/ and accept the Data Use Agreement.
2. Download the **ADNIMERGE2** R data package and the CSF/PET/plasma biomarker tables.
3. (Optional imaging arm) download the ADNI-1 baseline **T1 MRI** collection.

## 2. Expected layout

Place inputs so that scripts (run from the repo root) can find them:

```
<repo>/
  code/            # this repository
  data/            # you create this: ADNIMERGE2-derived tables + biomarker tables
  outputs/         # created by the scripts
  figures/         # created by revision_figures.py
```

## 3. Tabular pipeline (main text + supplement)

```bash
python code/extract_data.py
python code/build_manifest_cohorts.py
python code/build_features.py
python code/extract_biomarkers.py
python code/within_mci_staging.py          # -> outputs/within_mci/ (all classifiers; Table S1 inputs)
python code/fs_change_staging.py           # -> outputs/within_mci/fs_change_*.csv (Table S5a)
python code/amyloid_utility.py             # -> outputs/within_mci/amyloid_*.csv (Table 1, Figure 2, Tables S6-S7)
python code/revision_data.py               # -> data/amyloid_detail_baseline.csv, neuropath.csv, dxsum_detail.csv
python code/validation_repeated_apparent.py  # Table 2, Table S8, out-of-fold files
python code/validation_internal_external.py  # Table S9, Figure 4A (resumable)
python code/amyloid_cutoff_sensitivity.py  # Table S3, Figure 4B
python code/nonad_dementia_autopsy.py      # Tables S10-S11
python code/atrophy_redundancy.py          # Tables S5b, S12a, S12c
python code/atrophy_compact_robustness.py  # Table S12b
python code/fourgroup_compact_atrophy.py   # Table 2 (four-group compact row)
python code/age_matching_portable_model.py # Table S13
python code/table_s1_bootstrap.py          # Table S1
python code/revision_figures.py            # Figures 1-4
python code/export_frozen_models.py        # frozen ADNI portable models (external_validation/frozen_adni_models.json)
python code/portable_imaging_adni.py       # Table S15e (ADNI columns)
```

## 3b. External validation in OASIS-3 (Figure 5, Table S15)

Requires an approved OASIS-3 application (OASIS Data Use Agreement; only the authorized user may
access the data). Download the OASIS-3 tables from NITRC-IR (`0AS_data_files`: demographics, UDS B4 and
D1, amyloid Centiloid, PUP, FreeSurfer), then run locally:

```bash
python external_validation/oasis3_external_validation.py   # set OASIS_ROOT to the download folder
python code/figure5_external_validation.py
```

The script applies the frozen ADNI models unchanged and writes aggregate results only
(`external_validation/oasis_results/`; counts < 5 are masked in the log). MCI is defined by ADNI's
operational criteria (CDR 0.5 and MMSE >= 24); PiB PET is the primary amyloid tracer.

## 3c. External validation in NACC (Figure 5, Table S16)

Requires an approved NACC data request (non-commercial use; manuscripts must be submitted to NACC
before journal submission). Download the Quick-Access files (UDS investigator file; SCAN/CLARiTI and mixed-protocol
amyloid PET GAAIN files; SCAN MRI FreeSurfer file; SCAN PET QC file), then run locally:

```bash
python external_validation/nacc_external_validation.py   # set NACC_ROOT to the download folder
python code/figure5_external_validation.py
```

The script applies the frozen ADNI models unchanged and writes aggregate results only
(`external_validation/nacc_results/`; counts < 5 are masked). MoCA scores (UDS v3 onwards) are converted to
MMSE-equivalent scores with `external_validation/moca_mmse_crosswalk.csv` (Monsell et al., 2016, Table 3).

Reproduces: Tables 1-2, Figures 1-4 and Supplementary Tables S1-S3 and S5-S14 (S14 is a literature
summary); Figure 5 and Tables S15-S16 with sections 3b-3c. All revision analyses use the same fixed pipelines as `ml_common.py`, with imputation and
scaling fitted inside each training partition.

## 4. Optional imaging arm (Supplementary Table S4; GPU + raw T1)

```
% MATLAB (SPM12 + CAT12): run preproc/cat12_segment.m to segment the baseline T1 scans
python gpu_deep/build_staging_manifest.py   # staging manifest
python gpu_deep/staging_deep.py             # train 3D CNN -> leakage-safe fusion
```

Reproduces: Supplementary Table S4 (end-to-end 3D CNN within-MCI staging). The CNN is trained
from scratch; out-of-fold embeddings are fused with clinical/FreeSurfer features in a
fold-aligned, leakage-safe design (each subject predicted exactly once).

## Notes

- Hyper-parameters are fixed a priori (identical across feature families) for a fair, leakage-
  free comparison; the CV engine (`code/ml_common.py`) asserts subject-level train/test
  disjointness on every split.
- Randomness is seeded, but small numerical differences across platforms/library versions are
  expected and do not affect the qualitative conclusions.
