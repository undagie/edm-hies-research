# Dataset Specifications

## UCI Student Performance

- Expected input file: `data/raw/uci_student.csv`
- Target mapping: `at_risk = 1` when `G3 < 10`, else `0`
- Primary sensitive attribute: `sex` or `gender`
- Secondary sensitive attribute: `Medu` (fallback: `ses_proxy`)

## UCI Predict Students' Dropout and Academic Success

- Expected input file: `data/raw/uci_dropout_success.csv`
- Target mapping: `at_risk = 1` when `Target == Dropout` (case-insensitive)
- Primary sensitive attribute: `Gender` (fallback: `gender`/`sex`)
- Secondary sensitive attribute: `Scholarship holder` (fallback: `Nacionality`, then `ses_proxy`)

## xAPI-Edu-Data

- Expected input file: `data/raw/xapi.csv`
- Target mapping: `at_risk = 1` when `Class == L` (case-insensitive)
- Primary sensitive attribute: `gender`
- Secondary sensitive attribute: `NationalITy` (fallback: `ses_proxy`)

## Common Validation Rules

- Missing value policy:
  - Numeric -> median imputation
  - Categorical -> most frequent
- Split policy: stratified K-fold on `at_risk`
- Resampling policy: applied only on train fold
- Dataset availability policy:
  - Final/reproducibility runs: fail fast if raw CSV is missing
  - Development runs: synthetic fallback is allowed only via `--allow-synthetic-for-dev`

