# HIES: Heterogeneous Imbalance-Adaptive Ensemble (proposed comparators)

Two related methods are registered in `src/models/methods.py` and `configs/default.yaml`:

1. **`stacked_imbalance_ensemble_baseline`** — three heterogeneous, imbalance-adaptive base learners; a **logistic meta-learner** is fit on out-of-fold **class probabilities** (standard stacking). Bases: XGBoost with `scale_pos_weight`, `RandomForestClassifier` with `class_weight=balanced`, `HistGradientBoostingClassifier` with `class_weight=balanced`.

2. **`voting_imbalance_ensemble_baseline`** — the **same three bases** combined with **soft voting** (equal weights by default). Use this as an **ablation** of the fusion mechanism vs stacking.

**Suggested novelty framing (edit for your venue):** heterogeneous bases address complementary errors under class imbalance; stacking learns data-dependent fusion of probability estimates rather than a fixed vote.

**Suggested ablations in the manuscript:** (i) stacking vs voting; (ii) drop one base (e.g. RF-only vs full trio); (iii) sensitivity to `stack_cv` (e.g. 3 vs 5).

Hyperparameters live under `optimization.method_params` with nested keys `xgb`, `rf`, `hgb`, and for stacking `meta` and `stack_cv`.
