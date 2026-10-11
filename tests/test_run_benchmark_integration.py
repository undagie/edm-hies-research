from types import SimpleNamespace

import pandas as pd

from src.runner.experiment import (
    _apply_fairness_guardrails,
    _build_significance_table,
    _build_tradeoff_table,
    _compute_joint_score,
    _find_best_threshold,
    run_benchmark,
)


def test_run_benchmark_with_dev_synthetic_data():
    cfg = SimpleNamespace(
        seed=42,
        seeds=[42],
        n_splits=2,
        data={
            "raw_dir": "data/raw/does_not_need_to_exist_for_dev",
            "use_synthetic_if_missing": True,
            "dataset_order": ["xapi"],
        },
        methods=["logistic_regression"],
    )

    raw, summary = run_benchmark(cfg)

    assert not raw.empty
    assert not summary.empty
    assert "f1_macro" in raw.columns
    assert "f1_macro_mean" in summary.columns


def test_summary_ci_contains_mean_for_each_dataset_method():
    """f1_macro CI low/high brackets the row mean."""
    cfg = SimpleNamespace(
        seed=42,
        seeds=[42, 123],
        n_splits=2,
        data={
            "raw_dir": "data/raw/does_not_need_to_exist_for_dev",
            "use_synthetic_if_missing": True,
            "dataset_order": ["xapi"],
        },
        methods=["logistic_regression", "xgboost", "calibrated_xgboost"],
    )

    _, summary = run_benchmark(cfg)

    assert (summary["f1_macro_ci_low"] <= summary["f1_macro_mean"]).all()
    assert (summary["f1_macro_mean"] <= summary["f1_macro_ci_high"]).all()


def test_significance_table_uses_valid_non_approx_p_values():
    raw = pd.DataFrame(
        {
            "dataset": ["demo"] * 12,
            "seed": [42, 42, 123, 123, 2026, 2026] * 2,
            "fold": [0, 1, 0, 1, 0, 1] * 2,
            "method": ["xgboost"] * 6 + ["calibrated_xgboost"] * 6,
            "f1_macro": [0.61, 0.62, 0.64, 0.63, 0.60, 0.61, 0.60, 0.61, 0.63, 0.62, 0.59, 0.60],
        }
    )

    table = _build_significance_table(raw)

    assert "p_value" in table.columns
    assert "p_value_approx" not in table.columns
    assert ((table["p_value"] >= 0.0) & (table["p_value"] <= 1.0)).all()


def test_tradeoff_table_uses_seed_fold_paired_deltas():
    raw = pd.DataFrame(
        {
            "dataset": ["demo"] * 8,
            "seed": [42, 42, 123, 123, 42, 42, 123, 123],
            "fold": [0, 1, 0, 1, 0, 1, 0, 1],
            "method": ["xgboost"] * 4 + ["calibrated_xgboost"] * 4,
            "f1_macro": [0.70, 0.68, 0.72, 0.69, 0.69, 0.67, 0.70, 0.68],
            "demographic_parity_diff": [0.20, 0.19, 0.22, 0.21, 0.15, 0.16, 0.18, 0.17],
            "equal_opportunity_diff": [0.12, 0.11, 0.13, 0.12, 0.10, 0.10, 0.11, 0.10],
        }
    )

    opt = {"tradeoff_baseline_method": "xgboost", "tradeoff_target_method": "calibrated_xgboost"}
    table = _build_tradeoff_table(raw, opt)

    assert "paired_samples" in table.columns
    assert "delta_f1_macro_mean" in table.columns
    assert "delta_dp_diff_mean" in table.columns
    assert "fairness_win_rate" in table.columns
    assert "joint_win_rate" in table.columns
    assert table.loc[0, "paired_samples"] == 4
    assert table.loc[0, "delta_dp_diff_mean"] < 0


def test_compute_joint_score_penalizes_fairness_gaps():
    score_low_gap = _compute_joint_score(
        perf={"f1_macro": 0.8, "mcc": 0.6},
        fair={"demographic_parity_diff": 0.05, "equal_opportunity_diff": 0.04},
        weights={"f1_macro": 0.6, "mcc": 0.4, "dp_diff": 0.2, "eod_diff": 0.1},
    )
    score_high_gap = _compute_joint_score(
        perf={"f1_macro": 0.8, "mcc": 0.6},
        fair={"demographic_parity_diff": 0.25, "equal_opportunity_diff": 0.2},
        weights={"f1_macro": 0.6, "mcc": 0.4, "dp_diff": 0.2, "eod_diff": 0.1},
    )
    assert score_low_gap > score_high_gap


def test_find_best_threshold_returns_grid_value():
    y_true = pd.Series([1, 1, 0, 0, 1, 0]).to_numpy()
    y_prob = pd.Series([0.9, 0.8, 0.7, 0.4, 0.6, 0.2]).to_numpy()
    sensitive = pd.Series(["A", "A", "A", "B", "B", "B"]).to_numpy()

    best = _find_best_threshold(
        y_true=y_true,
        y_prob=y_prob,
        sensitive=sensitive,
        threshold_grid=[0.4, 0.5, 0.6],
        weights={"f1_macro": 0.7, "mcc": 0.3, "dp_diff": 0.2, "eod_diff": 0.1},
    )
    assert best in {0.4, 0.5, 0.6}


def test_apply_fairness_guardrails_flags_rows_against_baseline():
    raw = pd.DataFrame(
        {
            "dataset": ["demo", "demo", "demo", "demo"],
            "seed": [42, 42, 42, 42],
            "fold": [0, 0, 1, 1],
            "method": ["xgboost", "calibrated_xgboost", "xgboost", "calibrated_xgboost"],
            "demographic_parity_diff": [0.10, 0.12, 0.20, 0.23],
            "equal_opportunity_diff": [0.08, 0.09, 0.10, 0.12],
        }
    )
    out = _apply_fairness_guardrails(raw, max_dp_increase=0.03, max_eod_increase=0.02)
    cal_rows = out[out["method"] == "calibrated_xgboost"]
    assert cal_rows["fairness_guardrail_pass"].all()
