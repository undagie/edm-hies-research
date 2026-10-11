"""Stable method IDs vs manuscript-friendly labels (tables / LaTeX)."""

from __future__ import annotations

# Keys match ``results_summary.method``.
METHOD_MANUSCRIPT_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "calibrated_xgboost": "Calibrated XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "extra_trees": "Extra Trees",
    "random_forest": "Random Forest",
    "fairlearn_equalized_odds_xgb": "Fairlearn EO (XGBoost)",
    "tabpfn": "TabPFN",
    "voting": "Voting Ensemble",
    "stacking": "Stacking Ensemble",
    "stacking_rf_catboost": "RF-CatBoost Stacking",
    "hies": "HIES (proposed)",
}


def manuscript_label(method_id: str) -> str:
    """Return journal-style display name for a method ID."""
    return METHOD_MANUSCRIPT_LABELS.get(method_id, method_id.replace("_", " ").title())
