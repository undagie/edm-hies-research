# Dataset acquisition and preparation

The benchmark uses three publicly available Educational Data Mining datasets.
None are redistributed in this repository; you must download the raw CSVs
from the original sources and place them under `data/raw/`.

## Required files

```
data/raw/
├── uci_student.csv
├── uci_dropout_success.csv
└── xapi.csv
```

After placing the files, validate with:

```bash
python -m src.main --config configs/smoke.yaml
```

If a CSV is missing, the benchmark **fails fast** by design. Synthetic
fallback (`--allow-synthetic-for-dev`) exists only for development and is
never used for results reported in the paper.

## 1. UCI Student Performance

- **Source**: https://archive.ics.uci.edu/dataset/320/student+performance
- **File to place**: `data/raw/uci_student.csv`
- **Recommended source CSV**: `student-mat.csv` (Mathematics course; the
  Portuguese-course version is also valid, set with `bundle.frame` swap).
- **Target binarisation**: `target_at_risk = 1` if `G3 < 10`, else `0`.
  Threshold follows the conventional UCI EDM literature (50% passing grade).
- **Leakage columns dropped during training**: `G3`, `final_score`.
- **Primary sensitive attribute**: `sex` (mapped to `gender`).
- **Secondary sensitive attribute**: `Medu` (mother's education level).

## 2. UCI Predict Students' Dropout and Academic Success

- **Source**: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- **File to place**: `data/raw/uci_dropout_success.csv`
- **Target binarisation**: `target_at_risk = 1` if `Target == "Dropout"`
  (case-insensitive), else `0`. The original three-class problem (Dropout /
  Enrolled / Graduate) is binarised to the at-risk classification studied
  here.
- **Leakage columns dropped during training**: `Target`, `target`,
  `final_score`.
- **Primary sensitive attribute**: `Gender`.
- **Secondary sensitive attribute**: `Scholarship holder`.

## 3. xAPI-Edu-Data

- **Source**: https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data
- **File to place**: `data/raw/xapi.csv`
- **Target binarisation**: `target_at_risk = 1` if `Class == "L"`
  (Low achievement), else `0`.
- **Leakage columns dropped during training**: `Class`, `class`,
  `final_score`.
- **Primary sensitive attribute**: `gender`.
- **Secondary sensitive attribute**: `NationalITy`.

## Licensing and attribution

The datasets are not redistributed here; each remains under the terms of its
original source, and this repository's MIT licence covers only the code and the
generated analysis artifacts.

| Dataset | Terms | Cite as |
| --- | --- | --- |
| UCI Student Performance | UCI Machine Learning Repository, Creative Commons Attribution 4.0 | P. Cortez and A. M. G. Silva, "Using data mining to predict secondary school student performance", Proc. 5th Future Business Technology Conference, 2008 |
| UCI Predict Students' Dropout and Academic Success | UCI Machine Learning Repository, Creative Commons Attribution 4.0 | V. Realinho, J. Machado, L. Baptista, M. V. Martins, "Predicting student dropout and academic success", Data 7(11):146, 2022 |
| xAPI-Edu-Data | Kaggle dataset page terms; check the page before redistributing | E. A. Amrieh, T. Hamtini, I. Aljarah, "Mining educational data to predict student's academic performance using ensemble methods", Int. J. Database Theory and Application 9(8):119-136, 2016 |

Verify your downloads against the SHA-256 checksums recorded in
[`artifacts/manifests/PROVENANCE.md`](artifacts/manifests/PROVENANCE.md) before
reproducing. A mismatch means a different revision or export of the source
file, and the reported metrics will not reproduce exactly.

## Validation

The dataset summary written by the EDA script must produce these row counts:

| Dataset | N | Class 0 | Class 1 | Imbalance ratio |
| --- | --- | --- | --- | --- |
| uci_student            |  395 |  265 | 130  | 2.04 |
| uci_dropout_success    | 4424 | 3003 | 1421 | 2.11 |
| xapi                   |  480 |  353 | 127  | 2.78 |

If your counts differ, you have a different version of the source CSV.
Re-derive the target binarisation as documented above.

To run the validation:

```bash
python scripts/eda_datasets.py
```

This writes `artifacts/eda/dataset_summary.csv`. The row counts must match
the table above.

## Preprocessing applied at training time

All preprocessing happens inside the benchmark runner via
`src/models/methods.py::make_preprocessor`:

- Numeric columns: median imputation + `StandardScaler`.
- Categorical columns: most-frequent imputation + one-hot encoding
  (sklearn `OneHotEncoder(handle_unknown="ignore")`).
- Pairwise numeric feature interactions are appended when
  `data.enable_feature_interactions` is true (capped by
  `data.max_feature_interactions`, default 4).

The same `make_preprocessor` instance is fitted on each outer-CV training
fold and applied to the held-out test fold; preprocessor parameters are
**not** leaked across folds.
