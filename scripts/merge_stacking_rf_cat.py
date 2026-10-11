"""Merge RF-CatBoost stacking rows into canonical ``artifacts/results/``.

Reads ``artifacts_stack_rfcat/results/results_raw.csv``, recomputes fairness
guardrail vs XGBoost, appends via ``save_results``. Idempotent; backs up once.

Usage: python scripts/merge_stacking_rf_cat.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import load_config
from src.runner.experiment import save_results

METHOD = "stacking_rf_catboost"


def _recompute_guardrail(new_rows: pd.DataFrame, canonical: pd.DataFrame, opt_cfg: dict) -> pd.DataFrame:
    baseline = canonical[canonical["method"] == "xgboost"][
        ["dataset", "seed", "fold", "demographic_parity_diff", "equal_opportunity_diff"]
    ].rename(
        columns={
            "demographic_parity_diff": "baseline_dp_diff",
            "equal_opportunity_diff": "baseline_eod_diff",
        }
    )
    out = new_rows.drop(
        columns=["baseline_dp_diff", "baseline_eod_diff", "fairness_guardrail_pass"],
        errors="ignore",
    ).merge(baseline, on=["dataset", "seed", "fold"], how="left")
    out["fairness_guardrail_pass"] = (
        (out["demographic_parity_diff"] <= out["baseline_dp_diff"] + float(opt_cfg["max_dp_increase"]))
        & (out["equal_opportunity_diff"] <= out["baseline_eod_diff"] + float(opt_cfg["max_eod_increase"]))
    ).fillna(True)
    return out


def _summarise(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["dataset", "method"], as_index=False)
        .agg(
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            mcc_mean=("mcc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            bal_acc_mean=("balanced_accuracy", "mean"),
            dp_diff_mean=("demographic_parity_diff", "mean"),
            eod_diff_mean=("equal_opportunity_diff", "mean"),
            decision_threshold_mean=("decision_threshold", "mean"),
            joint_score_mean=("joint_score", "mean"),
            fairness_guardrail_pass_rate=("fairness_guardrail_pass", "mean"),
        )
        .sort_values(["dataset", "f1_macro_mean"], ascending=[True, False])
    )
    ci_rows: list[dict[str, float | str]] = []
    for (dataset_name, method_name), sub in raw.groupby(["dataset", "method"], sort=False):
        arr = sub["f1_macro"].to_numpy()
        if len(arr) > 1:
            res = bootstrap((arr,), np.mean, n_resamples=1000, random_state=0, method="basic")
            low = float(res.confidence_interval.low)
            high = float(res.confidence_interval.high)
        else:
            low = float(arr[0])
            high = float(arr[0])
        ci_rows.append(
            {
                "dataset": dataset_name,
                "method": method_name,
                "f1_macro_ci_low": low,
                "f1_macro_ci_high": high,
            }
        )
    return summary.merge(pd.DataFrame(ci_rows), on=["dataset", "method"], how="left")


def main() -> None:
    cfg = load_config("configs/default.yaml")
    canonical_path = Path("artifacts/results/results_raw.csv")
    new_path = Path("artifacts_stack_rfcat/results/results_raw.csv")
    canonical = pd.read_csv(canonical_path)
    new_raw = pd.read_csv(new_path)
    new_rows = new_raw[new_raw["method"] == METHOD].copy()
    if new_rows.empty:
        raise SystemExit(f"No {METHOD} rows found in {new_path}")

    expected = len(cfg.data["dataset_order"]) * len(cfg.seeds) * cfg.n_splits
    if len(new_rows) != expected:
        raise SystemExit(f"Expected {expected} rows for {METHOD}, found {len(new_rows)}")

    canonical = canonical[canonical["method"] != METHOD].copy()
    new_rows = _recompute_guardrail(new_rows, canonical, cfg.optimization)

    missing = set(canonical.columns) - set(new_rows.columns)
    extra = set(new_rows.columns) - set(canonical.columns)
    if missing or extra:
        print(f"Column mismatch — missing in new rows: {sorted(missing)}; extra: {sorted(extra)}")
    merged = pd.concat([canonical, new_rows[canonical.columns.intersection(new_rows.columns)]], ignore_index=True)
    merged = merged.sort_values(["dataset", "method", "seed", "fold"]).reset_index(drop=True)

    for name in ("results_raw.csv", "results_summary.csv"):
        src = Path("artifacts/results") / name
        bak = src.with_suffix(src.suffix + ".pre_stack_rfcat.bak")
        if not bak.exists():
            shutil.copy2(src, bak)

    summary = _summarise(merged)
    save_results(
        merged,
        summary,
        "artifacts",
        run_id="merge_stack_rfcat",
        optimization=cfg.optimization,
    )
    counts = merged.groupby("method").size()
    print(counts.to_string())
    print(f"Merged {len(new_rows)} {METHOD} rows into {canonical_path}")


if __name__ == "__main__":
    main()
