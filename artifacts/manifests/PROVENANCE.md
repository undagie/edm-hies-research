# Provenance of the shipped artifacts

This file records which artifact set was produced when, under which
configuration, and in which environment. It complements the per-run JSON
manifest and exists so that every number in the paper can be traced without
re-running the pipeline.

## Environment of record

The reported results were produced under:

| Component | Version |
| --- | --- |
| Python | 3.11.9 |
| scikit-learn | 1.4 |
| XGBoost | 2.0 |
| LightGBM | 4.3 |
| CatBoost | 1.2 |
| TabPFN | 2.0 |
| Fairlearn | 0.10 |

`requirements.txt` constrains these six libraries to exactly these minor
series. Installing newer releases will run the code but will not reproduce the
reported metrics to the third decimal, because scikit-learn, XGBoost and
Fairlearn changed relevant defaults after these versions.

Hardware: 16-core CPU, 64 GB RAM, Windows 11. All tree-based methods and HIES
train on CPU; only TabPFNv2 uses a GPU when one is available, falling back to
CPU otherwise.

## Input checksums

Verify before reproducing. These are the exact inputs behind the shipped
artifacts.

| File | SHA-256 | Bytes |
| --- | --- | --- |
| `configs/default.yaml` | `ffc77d7c424b6809a13bb69144b36495e73b1d146f15f99f2c0c412866c20cdc` | 4306 |
| `configs/stacking_rf_cat.yaml` | `7e6e5bd69c7de6608f2cfd9542436db87177f8a1705ba5cc27ce4c3e6cd9c792` | 2123 |
| `data/raw/uci_student.csv` | `235f58adaf87f909b6d9b423cdb48cc9b3d392e47b1ac7fd592d9a39fde3cc7d` | 42379 |
| `data/raw/uci_dropout_success.csv` | `a1b1a6531bbb93a5c7fdf0093b47172776652a4ffb12342fb79655c85b74801b` | 495642 |
| `data/raw/xapi.csv` | `49c8c699b5abadf599a30120e1e86a0ddff4a64946dbdae09c6198eb2165a69f` | 37545 |

The three dataset checksums are identical to those recorded in
`run_manifest_20260512T232607Z_bd43fb71.json`, so the input data has not
changed across any of the runs below.

## Artifact sets and their protocols

| Artifact set | Produced | Protocol | Rows in the raw file |
| --- | --- | --- | --- |
| `results/` (main benchmark, 12 methods) | 2026-07-16 | 12 seeds x 5 folds | 2160 = 12 x 3 x 12 x 5 |
| `phase1_analysis/table_significance_tost.csv` | 2026-07-16 | derived from `results/` | 33 comparisons |
| `ablation/` (six variants) | 2026-07-17 | 12 seeds x 5 folds | 1080 = 6 x 3 x 12 x 5 |
| `ablation_reduced_3x3/` (superseded) | 2026-07-16 | 3 seeds x 3 folds | 162 = 6 x 3 x 3 x 3 |
| `phase1_analysis/robustness_label_noise_raw.csv` | 2026-05-17 | 12 seeds x 5 folds, 4 noise levels | 2160 = 3 x 3 x 4 x 12 x 5 |
| `phase1_analysis/sensitivity_J_raw.csv` | 2026-05-10 | **3 seeds x 3 folds** | 972 = 3 x 9 x 4 x 3 x 3 |

Two protocol notes matter when reading these files.

The **ablation** was re-run at the full 12 x 5 protocol so that it is directly
comparable with the main benchmark. The earlier 3 x 3 run is retained under
`ablation_reduced_3x3/` for transparency only; it is superseded and should not
be quoted.

The **objective-weight sensitivity** study (paper Section 4.7) uses a reduced
3-seed by 3-fold protocol over 3 datasets, 9 weight settings and 4 methods.
This is a screening analysis: it asks only whether the ranking by the selection
score `J` is stable when each weight is perturbed by +/-50%, not what the
magnitude of the difference is. Its conclusion is a rank ordering that holds in
all 27 dataset-by-configuration combinations, which is why the lighter protocol
is sufficient. Do not compare its absolute values with the 12 x 5 numbers in
`results/`.

## The 12 May 2026 run manifest

`run_manifest_20260512T232607Z_bd43fb71.json` is the manifest of the
12 May 2026 main-benchmark run. It is kept as a historical record.

Its `config_sha256` field
(`d9e33a436540fe2dfc4c4f61a367e2aaca262d2d221d4dec71092b11e1c5ada7`) refers to
`configs/default.yaml` **as it stood on that date** and therefore does not
match the current file, whose checksum is given in the table above. The config
was subsequently updated and the main benchmark re-run on 16 July 2026, which
is the run behind the shipped `results/`. The dataset checksums in that
manifest are still valid, as verified above.

The manifest's `environment` block records the interpreter but not the
installed package versions, and its `git` block was written before the
repository had a commit history. The environment table at the top of this file
supersedes both.

## Fairness and the decision threshold

The HIES estimator and the experiment runner both contain optional code paths
that can let fairness influence threshold selection: the
`fairness_threshold_penalty` term in `src/models/methods.py` and the
`guardrail_fn` argument of `find_group_thresholds` in
`src/runner/experiment.py`.

**Both are disabled in every shipped configuration.** `configs/default.yaml`
sets `fairness_threshold_penalty: 0.0`, and `use_group_thresholds` is never
enabled, so it keeps its default of `false`. The decision threshold is
therefore selected on the joint F1 + MCC objective alone, matching the
manuscript, where fairness enters only the selection score `J` used for ranking
complete methods and never the choice of the threshold.

`artifacts/tables/table_guardrail.csv` reports fairness-guardrail pass rates as
a diagnostic over the results; it does not feed back into model fitting or
threshold selection.
