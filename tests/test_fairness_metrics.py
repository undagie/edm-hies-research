import numpy as np

from src.evaluation.metrics import compute_fairness


def test_compute_fairness_supports_multigroup_range_gap():
    """DP/EOD are max-minus-min gaps across three sensitive groups."""
    y_true = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1])
    sensitive = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])

    metrics = compute_fairness(y_true, y_pred, sensitive)

    assert metrics["demographic_parity_diff"] == 1 / 3
    assert metrics["equal_opportunity_diff"] == 1 / 2
