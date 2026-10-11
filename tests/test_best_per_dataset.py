import pandas as pd

from src.reporting.best_per_dataset import select_best_per_dataset


def test_select_best_per_dataset_primary_and_tiebreak():
    """Per-dataset best uses f1_macro then joint_score tiebreak."""
    summary = pd.DataFrame(
        [
            {"dataset": "a", "method": "m1", "f1_macro_mean": 0.9, "joint_score_mean": 1.0, "mcc_mean": 0.5},
            {"dataset": "a", "method": "m2", "f1_macro_mean": 0.91, "joint_score_mean": 0.9, "mcc_mean": 0.6},
            {"dataset": "a", "method": "m3", "f1_macro_mean": 0.91, "joint_score_mean": 1.1, "mcc_mean": 0.4},
            {"dataset": "b", "method": "m1", "f1_macro_mean": 0.8, "joint_score_mean": 1.0, "mcc_mean": 0.5},
        ]
    )
    best = select_best_per_dataset(summary, exclude_methods=())
    assert len(best) == 2
    row_a = best[best["dataset"] == "a"].iloc[0]
    assert row_a["method"] == "m3"
    row_b = best[best["dataset"] == "b"].iloc[0]
    assert row_b["method"] == "m1"


def test_dummy_excluded_by_default():
    summary = pd.DataFrame(
        [
            {"dataset": "a", "method": "dummy_most_frequent_baseline", "f1_macro_mean": 1.0, "joint_score_mean": 9.0},
            {"dataset": "a", "method": "xgboost", "f1_macro_mean": 0.5, "joint_score_mean": 1.0},
        ]
    )
    best = select_best_per_dataset(summary)
    assert best.iloc[0]["method"] == "xgboost"
