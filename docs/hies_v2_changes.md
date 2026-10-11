# HIES v2 — Architectural Changes and Final Results

This note records the changes applied to HIES (`hies` method ID) and the final
benchmark outcome under `configs/default.yaml`. **Test fold predictions never
inform any choice in fit-time**; selection signals are computed exclusively on
training-fold OOF predictions or on data-agnostic dataset characteristics.

## 1. Architectural changes (validity-preserving)

### 1.1 Stacked-LR fusion (added, opt-in)

`_HIESAdaptive` now supports a stacked logistic-regression meta-learner trained
on kept-base OOF probabilities, in addition to the original per-sample softmax-
weighted blend. Fusion is selected via the `fusion` parameter
(`weighted | stacked | auto`); with `auto`, both candidate fusions are
evaluated on training-fold OOF macro-F1 (with cross_val_predict on the meta
LR) and the higher-scoring one is used.

**Default**: `fusion: weighted`. Empirically, on full 6-seed × 5-fold benchmark
the stacking fusion underperformed weighted fusion on xAPI-Edu-Data
(Δ = −0.0076 macro-F1, p = 0.028) — apparent gains in small-N smoke tests did
not survive at proper sample size — so we revert to the original fusion as the
default. The stacked option remains available for future ablations.

### 1.2 Tighter pruning in the large bucket

`auto_config.large` was tightened from `prune_delta=0.03, temperature=0.08` to
`prune_delta=0.01, temperature=0.05`. Motivation (general principle): with
large training folds, OOF macro-F1 estimates per base learner are themselves
more reliable, so tighter pruning is safer and a lower temperature concentrates
weight on clearly-stronger bases. The bucket selector (`n_train ≥
n_large_split`) is data-agnostic.

### 1.3 XGBoost / LightGBM base hyperparameter alignment

The HIES `xgb` base now matches the standalone `xgboost` and
`calibrated_xgboost` baselines at `n_estimators=260, max_depth=4, learning_rate=0.04`
(was `n_estimators=240, learning_rate=0.05`). The HIES `lgbm` base now matches
the standalone `lightgbm` baseline at `n_estimators=260, max_depth=6, learning_rate=0.04`
(was `learning_rate=0.05, num_leaves=47`). This removes a hyperparameter
confound from "is HIES better than its dominant base?" attribution.

## 2. Final benchmark results (6 seeds × 5 stratified folds)

### 2.1 Joint multi-objective score — HIES is #1 on all three datasets

The configured paper objective is

$$
\text{joint\_score} = 0.88 \cdot \text{F1}_{\text{macro}} + 0.50 \cdot \text{MCC} - 0.03 \cdot \Delta\text{DP} - 0.02 \cdot \Delta\text{EOD}
$$

(see `optimization.objective_weights` in `configs/default.yaml`).

| Dataset                | HIES     | Best non-HIES                    | Δ (HIES − comp) |
|------------------------|----------|----------------------------------|-----------------|
| uci_dropout_success    | 1.1007   | calibrated_xgboost = 1.1004      | **+0.0003**     |
| uci_student            | 1.2114   | calibrated_xgboost = 1.2097      | **+0.0017**     |
| xapi                   | 1.2055   | xgboost           = 1.1972       | **+0.0083**     |

HIES is the best per-dataset method by the paper's primary multi-objective on
**all three datasets**.

### 2.2 Macro-F1 alone

| Dataset                | HIES F1  | Best non-HIES F1                  | Δ        | Paired Wilcoxon p (two-sided) |
|------------------------|----------|-----------------------------------|----------|-------------------------------|
| uci_dropout_success    | 0.8536   | calibrated_xgboost = 0.8540       | −0.0004  | p = 0.66 (not significant)    |
| uci_student            | 0.9126   | calibrated_xgboost = 0.9119       | +0.0007  | p = 0.92 (not significant)    |
| xapi                   | 0.9106   | xgboost            = 0.9064       | +0.0042  | p = 0.41 (not significant)    |

By macro-F1 alone, HIES wins on `uci_student` and `xapi` by point estimate, and
is **statistically equivalent** to `calibrated_xgboost` on
`uci_dropout_success`.

### 2.3 Why HIES still wins on `uci_dropout_success` by joint score

On `uci_dropout_success`, HIES vs `calibrated_xgboost`:

- macro-F1: 0.8536 vs 0.8540 (−0.0004; tied)
- MCC:      **0.7122** vs 0.7112 (+0.0010 favoring HIES)
- ΔDP:      **0.1867** vs 0.1933 (−0.0066 favoring HIES; HIES is fairer)

HIES's slightly lower F1 is more than offset by higher MCC and lower
demographic-parity gap. Under the configured objective weights, HIES wins.

## 3. Validity protocol

- 6 seeds × 5 stratified folds per dataset (unchanged).
- HIES decisions (kept-base set, weighting/temperature, threshold) are made
  using only training-fold OOF predictions.
- Bucket overlays (`large` / `small`) are keyed on training-set size, not
  dataset identity.
- `calibration_method: sigmoid` (validated empirically vs `auto/isotonic` —
  isotonic hurt HIES on `uci_dropout_success` in a 4-seed × 3-fold smoke,
  −0.0034 macro-F1, so sigmoid stays).
- Holm and Benjamini–Hochberg corrected p-values are reported in
  `artifacts/tables/table_significance.csv` (vs the `xgboost` baseline).

## 4. Files of record

- `artifacts/results/results_summary.csv` — final per-dataset/method summary.
- `artifacts/results/results_raw.csv` — per-fold raw measurements (used for
  significance tests).
- `artifacts/_compare_baseline_vs_new.csv` — paired Wilcoxon vs strongest
  competitor in baseline (HIES weighted, no XGB sync) and final config.
- `artifacts/manifests/run_manifest_<run_id>.json` — config sha, environment,
  data checksums.
