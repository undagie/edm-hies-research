"""Performance, calibration, and fairness metrics for the benchmark."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def compute_performance(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Equal-width-bin Expected Calibration Error."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        avg_p = float(y_prob[mask].mean())
        avg_y = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(avg_p - avg_y)
    return float(ece)


def compute_calibration(y_true, y_proba) -> dict[str, float]:
    """Brier score and 10-bin ECE."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    return {
        "brier_score": float(brier_score_loss(y_true_arr, y_proba_arr)),
        "ece": _expected_calibration_error(y_true_arr, y_proba_arr, n_bins=10),
    }


def compute_fairness(y_true, y_pred, sensitive) -> dict[str, float]:
    """DP / EoD / PP as max−min group gaps (0.0 if fewer than two groups)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    sensitive = np.asarray(sensitive)
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return {
            "demographic_parity_diff": 0.0,
            "equal_opportunity_diff": 0.0,
            "predictive_parity_diff": 0.0,
        }
    selection_rates: list[float] = []
    tpr_rates: list[float] = []
    ppv_rates: list[float] = []
    for group in groups:
        mask = sensitive == group
        if not mask.any():
            continue
        selection_rates.append(float(y_pred[mask].mean()))
        tpr_rates.append(
            float(recall_score(y_true[mask], y_pred[mask], zero_division=0))
        )
        # PPV undefined when a group has no predicted positives.
        if y_pred[mask].sum() > 0:
            ppv_rates.append(
                float(precision_score(y_true[mask], y_pred[mask], zero_division=0))
            )
    if len(selection_rates) < 2 or len(tpr_rates) < 2:
        return {
            "demographic_parity_diff": 0.0,
            "equal_opportunity_diff": 0.0,
            "predictive_parity_diff": 0.0,
        }
    out = {
        "demographic_parity_diff": float(max(selection_rates) - min(selection_rates)),
        "equal_opportunity_diff": float(max(tpr_rates) - min(tpr_rates)),
    }
    if len(ppv_rates) >= 2:
        out["predictive_parity_diff"] = float(max(ppv_rates) - min(ppv_rates))
    else:
        out["predictive_parity_diff"] = 0.0
    return out


def compute_calibration_per_group(y_true, y_proba, sensitive, n_bins: int = 10) -> dict[str, float]:
    """Per-group ECE max−min spread (calibration parity)."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    sensitive = np.asarray(sensitive)
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return {"calibration_parity_diff": 0.0}
    eces = []
    for group in groups:
        mask = sensitive == group
        if mask.sum() < 5:
            continue
        eces.append(_expected_calibration_error(y_true_arr[mask], y_proba_arr[mask], n_bins=n_bins))
    if len(eces) < 2:
        return {"calibration_parity_diff": 0.0}
    return {"calibration_parity_diff": float(max(eces) - min(eces))}

