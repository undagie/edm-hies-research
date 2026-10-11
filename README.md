# Heterogeneous Imbalance-Adaptive Ensemble for Robust Prediction of Students at Risk on Tabular Educational Data

Reference implementation and reproduction package for **HIES**, a
Heterogeneous Imbalance-adaptive Ensemble System.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible benchmark and reference implementation for **HIES**, a
heterogeneous imbalance-adaptive ensemble that combines per-base calibrated
probabilities via out-of-fold (OOF) pruning, per-sample confidence weighting,
and an inner-CV-tuned decision threshold.

The threshold is tuned on a joint macro-F1 + MCC objective only. Fairness is
audited, not optimised: the demographic-parity and equal-opportunity
differences enter the selection score `J` that ranks complete methods, and
never the choice of the decision threshold. See
[`artifacts/manifests/PROVENANCE.md`](artifacts/manifests/PROVENANCE.md) for
the configuration flags that hold this guarantee.

HIES is benchmarked against modern tabular SOTA (TabPFN), classical baselines
(XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Logistic
Regression), ensemble baselines (Soft Voting, Stacking with LR meta,
RF-CatBoost Stacking), and a fairness-targeted post-processor (Fairlearn
Equalized Odds) on three Educational Data Mining datasets.

## Headline result

After fair benchmarking (12 random seeds × 5-fold stratified CV = 60 paired
fold samples per method, identical CatBoost iteration budgets across
ensembles, no Calibrated-XGBoost double-counting), **HIES achieves the
highest macro-F1 on all three datasets**:

| Dataset                  | HIES F1 | #2 method              | Holm vs XGB              | TOST non-inferior      |
| ------------------------ | ------- | ---------------------- | ------------------------ | ---------------------- |
| UCI Student Performance  | 0.9098  | XGBoost   (+0.0011)    | p = 1.0, not significant | yes, p = 0.0012        |
| UCI Dropout & Success    | 0.8552  | Voting    (+0.0014)    | p = 8.2e-4, significant  | yes, p = 0 (underflow) |
| xAPI-Edu-Data            | 0.9157  | CatBoost  (+0.0021)    | p = 3.5e-4, significant  | yes, p = 3.3e-9        |

HIES is also Holm-significantly better than XGBoost on two of three datasets;
on the third (UCI Student Performance) HIES wins nominally and is **TOST
non-inferior** to all baselines at margin 0.005 macro-F1.

## Project layout

```
edm-hies-research/
├── configs/                 # Single YAML configuration drives the benchmark
│   └── default.yaml
├── data/                    # Place raw CSVs under data/raw/ (see DATASETS.md)
├── src/                     # Library code (do not duplicate logic in scripts)
│   ├── config.py            # Config loader (dataclass)
│   ├── datasets/loader.py   # Dataset bundles + target binarisation
│   ├── evaluation/metrics.py# Performance / fairness / calibration metrics
│   ├── models/methods.py    # Method factory + HIES estimator
│   ├── reporting/           # Method labels, best-per-dataset, figure gen
│   ├── runner/experiment.py # Benchmark loop with shared seeds + CV
│   ├── tracking.py          # MLflow + run manifest helpers
│   └── main.py              # CLI entry point
├── scripts/                 # Analysis pipelines that consume artifacts/
├── tests/                   # pytest sanity + integration tests
├── docs/                    # Design notes, dataset specs, validity report
├── artifacts/               # Generated outputs (key result CSVs, tables, and
│                            #   figures are tracked so every paper number is
│                            #   independently verifiable; logs are ignored)
└── DATASETS.md              # Dataset acquisition + preparation instructions
```

## Quick start (smoke run, ~5 minutes)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place the three raw CSVs under data/raw/ (see DATASETS.md)
python -m src.main --config configs/smoke.yaml
```

Smoke output lands under `artifacts_smoke/`.

`requirements.txt` pins scikit-learn, XGBoost, LightGBM, CatBoost, TabPFN and
Fairlearn to the minor series that produced the reported numbers (Python
3.11.9). Newer releases of those libraries run the code but shift the metrics
in the third decimal. The full environment record, input checksums, and the
protocol behind each artifact set are in
[`artifacts/manifests/PROVENANCE.md`](artifacts/manifests/PROVENANCE.md).

## Full reproduction (~3 hours on a workstation with one GPU)

The full benchmark is one command:

```bash
python scripts/reproduce_full.py
```

`reproduce_full.py` runs the canonical pipeline end-to-end:

1. Main benchmark on the three datasets (configs/default.yaml, 11 methods).
2. Phase-1 analyses: TOST non-inferiority with Holm correction, bootstrap
   CIs and Cliff's delta effect sizes, per-group F1/TPR/FPR/PPV subgroup
   metrics, utility table (students per 1,000), fairness-utility Pareto
   plot, paired-difference boxplot, calibration table.
3. Friedman omnibus + Nemenyi post-hoc + average-rank/CD diagrams.
4. SHAP top-15 global features per dataset + per-sensitive-group SHAP.

The reviewer-requested RF-CatBoost stacking baseline (12th method) runs from
its own config and is merged into the canonical results:

```bash
python -m src.main --config configs/stacking_rf_cat.yaml   # 12 seeds x 5 folds
python scripts/merge_stacking_rf_cat.py                    # merge + regenerate tables
python scripts/analyze_phase1.py                           # refresh stats incl. new method
python scripts/analyze_friedman_nemenyi.py                 # refresh ranks / Fig. 2
```

This covers paper Tables 4–6 and Fig. 2. The remaining studies are
**separate commands** (not orchestrated by `reproduce_full.py`):

```bash
python scripts/run_ablation.py                                   # Table 7 (ablation, full 12x5; --reduced for legacy 3x3)
python scripts/run_sensitivity_robustness.py                     # Section 4.7 weight sensitivity (screening, 3x3)
python scripts/run_sensitivity_robustness.py --only noise --full # Table 8 + Fig. 3 data (full 12x5)
python scripts/generate_robustness_plot.py                       # Figure 3
python scripts/analyze_label_noise.py                            # Table 8 + Section 4.8 Wilcoxon p-values
python scripts/verify_threshold_route.py                         # Section 4.4 operating-point check (Table 7 `full` vs Table 4)
python scripts/generate_shap_figure.py                           # Figure 4
python scripts/generate_hies_architecture.py                     # Figure 1
```

The weight-sensitivity study is the one analysis that does **not** use the
12-seed by 5-fold protocol; it screens the stability of the `J` ranking over
3 seeds by 3 folds. The reasoning is recorded in
[`artifacts/manifests/PROVENANCE.md`](artifacts/manifests/PROVENANCE.md).

Expected wall-clock on a single workstation (12-core CPU, 1× NVIDIA GPU):
~150 minutes for the main benchmark, ~10 minutes for the downstream
analyses, and ~7 hours for the full-protocol label-noise study. All outputs
go under `artifacts/` and `artifacts/phase1_analysis/`.

## Reproducing the paper tables and figures

After `scripts/reproduce_full.py`, the paper-relevant outputs are:

| Paper artifact | Source |
| --- | --- |
| Table 1 (datasets + subgroup sizes) | Descriptive — §3.1 + `artifacts/manifests/PROVENANCE.md` (dataset SHA-256) |
| Tables 2–3 (hyperparams, buckets)| `configs/default.yaml` |
| Table 4 (macro-F1 / MCC)        | `artifacts/results/results_summary.csv` |
| Table 5 (fairness & calibration)| `artifacts/results/results_summary.csv` + `artifacts/phase1_analysis/table_calibration.csv` |
| Table 6 (Wilcoxon + TOST + CI + Cliff's delta) | `artifacts/phase1_analysis/table_significance_tost.csv` (+ `table_utility.csv` for the per-1,000 reading) |
| Table 7 (ablation, full 12×5)   | `artifacts/ablation/table_ablation_f1.csv` (legacy reduced-protocol run preserved under `artifacts/ablation_reduced_3x3/`) |
| Table 7 operating-point check   | `artifacts/ablation/verify_threshold_route.csv` — per-fold proof that the Table 7 `full` row and the Table 4 HIES row come from one fitted model scored at two operating points (`scripts/verify_threshold_route.py`) |
| Table 8 (label noise)           | `artifacts/phase1_analysis/robustness_label_noise_summary.csv` (+ `scripts/analyze_label_noise.py` for the §4.6 Wilcoxon p-values) |
| Table 9 (literature positioning)| Manual — refs [2], [3], [25]; no computed artifact |
| Table 10 (notation)             | Descriptive — Notation List section; no computed artifact |
| Figure 1 (architecture)         | `artifacts/diagrams/hies_architecture.pdf` |
| Figure 2 (average ranks + CD)   | `artifacts/phase1_analysis/cd_diagram_per_dataset.pdf` |
| Figure 3 (label noise)          | `artifacts/phase1_analysis/robustness_label_noise.pdf` |
| Figure 4 (SHAP, UCI Dropout)    | `artifacts/phase1_analysis/shap_top15_uci_dropout_success.pdf` |
| Supplementary (Pareto, per-group SHAP) | `artifacts/phase1_analysis/pareto_f1_dp.pdf`, `shap_per_group_top10_*.csv` |

Every table above is a rounded view of a full-precision CSV; all deltas
quoted in the paper are computed from the full-precision per-seed, per-fold
values in `artifacts/results/results_raw.csv` and
`artifacts/phase1_analysis/*_raw.csv` before rounding.

Section 4.7 (objective-weight sensitivity) is backed by
`artifacts/phase1_analysis/sensitivity_J_summary.csv`; the 27
dataset-by-configuration combinations it reports come from the raw file of the
same name with the `_raw` suffix.

## Methodological choices

- Why `Calibrated XGBoost` is **not** in the main comparison
  (HIES already integrates calibrated XGBoost as its `xgb_spw` base; listing
  Calibrated XGBoost as a separate baseline double-counts a component of
  HIES — this follows the AutoML / tabular SOTA convention).
- Why per-base hyperparameters are aligned across HIES, Voting, and Stacking
  (CatBoost iterations are identically set to 400, etc.).
- Why the threshold-tuning protocol is conservative for HIES.
- TabPFN subsampling policy on the largest dataset.
- Multiple-comparisons correction (Holm, BH, Friedman+Nemenyi, TOST).
- Cross-validation protocol and hyperparameter selection.

## Citation

If you use HIES or our experimental protocol, please cite:

```bibtex
@article{firdaus2026hies,
  title   = {Heterogeneous Imbalance-adaptive Ensemble for Robust Prediction of Students at Risk on Tabular Educational Data},
  author  = {Firdaus, Muhammad Iqbal and Rosadi, Muhammad Edya and Agustini, Dian and Muflih, M and Dewi, Deshinta Arrova},
  journal = {International Journal of Intelligent Engineering and Systems},
  volume  = {19},
  number  = {10},
  year    = {2026},
  doi     = {10.22266/ijies2026.1031.08},
  url     = {https://inass.org/wp-content/uploads/2026/06/2026103108-2.pdf}
}
```

Code and reproduction package: https://github.com/undagie/edm-hies-research

A `CITATION.cff` is included for tools that resolve software citations
automatically.

## License

MIT — see [`LICENSE`](LICENSE).
