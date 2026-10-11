import pandas as pd

from src.runner.experiment import _prepare_xy


def test_prepare_xy_drops_target_sensitive_and_leakage_columns():
    df = pd.DataFrame(
        {
            "feat1": [1.0, 2.0, 3.0],
            "final_score": [9.0, 14.0, 6.0],
            "target_at_risk": [1, 0, 1],
            "gender": ["F", "M", "F"],
        }
    )
    X, y, sensitive = _prepare_xy(
        df,
        target_col="target_at_risk",
        sensitive_col="gender",
        leakage_columns=["final_score"],
    )

    assert list(X.columns) == ["feat1"]
    assert y.tolist() == [1, 0, 1]
    assert sensitive.tolist() == ["F", "M", "F"]


def test_prepare_xy_adds_limited_feature_interactions_without_leakage():
    df = pd.DataFrame(
        {
            "feat1": [1.0, 2.0, 3.0],
            "feat2": [0.5, 0.1, 0.2],
            "final_score": [9.0, 14.0, 6.0],
            "target_at_risk": [1, 0, 1],
            "gender": ["F", "M", "F"],
        }
    )
    X, _, _ = _prepare_xy(
        df,
        target_col="target_at_risk",
        sensitive_col="gender",
        leakage_columns=["final_score"],
        enable_feature_interactions=True,
        max_interactions=1,
    )
    assert "final_score" not in X.columns
    assert "feat1_x_feat2" in X.columns
