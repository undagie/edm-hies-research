"""Generate robustness sensitivity and extended-seed tables."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.runner.experiment import _build_significance_table, _build_tradeoff_table, finalize_mlflow, run_benchmark
from src.utils import silence_known_warnings

silence_known_warnings()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness analysis tables.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def _delta_table(summary: pd.DataFrame, profile_name: str, baseline_method: str, target_method: str) -> list[dict]:
    rows: list[dict] = []
    for dataset, sub in summary.groupby("dataset"):
        by_method = {row["method"]: row for _, row in sub.iterrows()}
        base = by_method.get(baseline_method)
        targ = by_method.get(target_method)
        if base is None or targ is None:
            continue
        rows.append(
            {
                "profile": profile_name,
                "dataset": dataset,
                "delta_f1_macro_mean": float(targ["f1_macro_mean"] - base["f1_macro_mean"]),
                "baseline_f1_macro_mean": float(base["f1_macro_mean"]),
                "target_f1_macro_mean": float(targ["f1_macro_mean"]),
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    tables_dir = Path(cfg.outputs["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    opt = cfg.optimization if isinstance(cfg.optimization, dict) else {}
    baseline_method = str(opt.get("tradeoff_baseline_method", "xgboost"))
    target_method = str(opt.get("tradeoff_target_method", "calibrated_xgboost"))

    # Disable MLflow tracking for robustness batch to avoid external lock contention.
    cfg.outputs = None

    profiles = {
        "utility_high": {"f1_macro": 0.86, "mcc": 0.48, "dp_diff": 0.04, "eod_diff": 0.03},
        "balanced_default": {"f1_macro": 0.82, "mcc": 0.45, "dp_diff": 0.06, "eod_diff": 0.05},
        "fairness_heavier": {"f1_macro": 0.76, "mcc": 0.4, "dp_diff": 0.1, "eod_diff": 0.08},
    }
    sensitivity_rows: list[dict] = []
    for profile_name, weights in profiles.items():
        cfg_p = deepcopy(cfg)
        cfg_p.optimization = deepcopy(cfg.optimization)
        cfg_p.optimization["objective_weights"] = weights
        _, summary_p = run_benchmark(cfg_p, run_id=f"robustness_{profile_name}")
        finalize_mlflow()
        sensitivity_rows.extend(_delta_table(summary_p, profile_name, baseline_method, target_method))
    pd.DataFrame(sensitivity_rows).to_csv(tables_dir / "table_robustness_sensitivity.csv", index=False)

    cfg_s = deepcopy(cfg)
    cfg_s.seeds = [11, 22, 33, 44, 55, 66]
    raw_s, _ = run_benchmark(cfg_s, run_id="robustness_extended_seed")
    finalize_mlflow()

    seed_significance = _build_significance_table(raw_s)
    seed_tradeoff = _build_tradeoff_table(raw_s, cfg_s.optimization if isinstance(cfg_s.optimization, dict) else {})
    seed_significance.to_csv(tables_dir / "table_robustness_seed_significance.csv", index=False)
    seed_tradeoff.to_csv(tables_dir / "table_robustness_seed_tradeoff.csv", index=False)


if __name__ == "__main__":
    main()
