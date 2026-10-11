"""HIES ablation: six variants on 3 datasets (default 12×5; ``--reduced`` = 3×3).

Variants: full, no_pruning, no_per_sample, no_threshold, stacked, no_calibration.
Writes ``artifacts/ablation/results_*.csv`` and pivoted F1/joint tables.

Usage: python scripts/run_ablation.py [--reduced]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import load_config
from src.datasets import load_dataset_bundle
from src.evaluation import (
    compute_calibration,
    compute_calibration_per_group,
    compute_fairness,
    compute_performance,
)
from src.models.methods import (
    _AutoCalibratedEstimator,
    _HIESAdaptive,
    _heterogeneous_calibrated_bases,
    build_method,
    make_preprocessor,
)
from src.runner.experiment import _compute_joint_score, _prepare_xy
from src.utils import silence_known_warnings

silence_known_warnings()


def _make_full_estimator(seed: int, class_weight_ratio: float, mp: dict):
    """Standard HIES build path so ``full`` matches the main benchmark."""
    spec = build_method(
        "hies",
        seed=seed,
        class_weight_ratio=class_weight_ratio,
        method_params={"hies": deepcopy(mp)},
    )
    return spec.estimator


def _make_no_calibration_estimator(seed: int, class_weight_ratio: float, mp: dict):
    """Same HIES wiring with raw (uncalibrated) base learners."""
    cfg = deepcopy(mp)
    xgb_kw = dict(cfg.pop("xgb", {}))
    rf_kw = dict(cfg.pop("rf", {}))
    hgb_kw = dict(cfg.pop("hgb", {}))
    lgbm_kw = dict(cfg.pop("lgbm", {}))
    bases = _heterogeneous_calibrated_bases(
        seed=seed,
        class_weight_ratio=class_weight_ratio,
        xgb_params=xgb_kw,
        rf_params=rf_kw,
        hgb_params=hgb_kw,
        lgbm_params=lgbm_kw,
        calibration_cv=int(cfg.get("calibration_cv", 3)),
        calibration_method="sigmoid",
        n_train_threshold=int(cfg.get("n_train_threshold", 1500)),
    )
    raw_bases = [
        (name, base.estimator) if isinstance(base, _AutoCalibratedEstimator) else (name, base)
        for name, base in bases
    ]
    auto_cfg = dict(cfg.pop("auto_config", {}))
    auto_enabled = bool(auto_cfg.pop("enabled", False))
    return _HIESAdaptive(
        estimators=raw_bases,
        prune_delta=float(cfg.get("prune_delta", 0.025)),
        min_keep=int(cfg.get("min_keep", 3)),
        temperature=float(cfg.get("temperature", 0.08)),
        margin_floor=float(cfg.get("margin_floor", 0.05)),
        cv_splits=int(cfg.get("cv_splits", 5)),
        tune_threshold=bool(cfg.get("tune_threshold", True)),
        threshold_grid=list(cfg.get("threshold_grid", [0.40, 0.43, 0.46, 0.50, 0.54])),
        threshold_clip=tuple(cfg.get("threshold_clip", [0.30, 0.70])),
        objective_weights=cfg.get(
            "objective_weights",
            {"f1_macro": 0.88, "mcc": 0.50, "dp_diff": 0.03, "eod_diff": 0.02},
        ),
        auto_config=auto_enabled,
        auto_n_large=int(auto_cfg.get("n_large_split", 1500)),
        auto_imb_thresh=float(auto_cfg.get("imbalance_split", 2.5)),
        auto_small=auto_cfg.get("small"),
        auto_large=auto_cfg.get("large"),
        auto_feature_dim_split=auto_cfg.get("feature_dim_split"),
        auto_wide=auto_cfg.get("wide"),
        auto_narrow=auto_cfg.get("narrow"),
        auto_imbalanced_mcc_boost=float(auto_cfg.get("imbalanced_mcc_boost", 0.0)),
        fusion=str(cfg.get("fusion", "weighted")),
        random_state=seed,
    )


def _variant_params(name: str, base_mp: dict) -> dict:
    mp = deepcopy(base_mp)
    if name == "full":
        return mp
    if name == "no_pruning":
        # Huge prune_delta keeps all bases on both size buckets.
        for bucket in ("small", "large"):
            mp.setdefault("auto_config", {}).setdefault(bucket, {})["prune_delta"] = 1.0
        return mp
    if name == "no_per_sample":
        # Inflated margin_floor collapses weights to global skill weights only.
        mp["margin_floor"] = 1e6
        return mp
    if name == "no_threshold":
        mp["tune_threshold"] = False
        return mp
    if name == "stacked":
        mp["fusion"] = "stacked"
        # Bucket presets otherwise override top-level fusion back to weighted.
        for sub in ("wide", "narrow"):
            mp.setdefault("auto_config", {}).setdefault(sub, {}).pop("fusion", None)
        for bucket in ("small", "large"):
            mp.setdefault("auto_config", {}).setdefault(bucket, {})["fusion"] = "stacked"
        return mp
    if name == "no_calibration":
        return mp  # removed via custom estimator builder
    raise ValueError(f"Unknown variant: {name}")


def evaluate_variant(name: str, cfg, seeds: list[int], n_splits: int) -> list[dict]:
    rows: list[dict] = []
    base_mp = deepcopy(cfg.optimization["method_params"]["hies"])
    # No runner here: enable OOF threshold tuning so no_threshold ≠ full.
    base_mp["tune_threshold"] = True
    mp = _variant_params(name, base_mp)
    weights = base_mp.get("objective_weights", cfg.optimization["objective_weights"])
    for ds in cfg.data["dataset_order"]:
        bundle = load_dataset_bundle(
            ds,
            cfg.data["raw_dir"],
            seed=cfg.seed,
            allow_synthetic=False,
        )
        X, y, sens = _prepare_xy(
            bundle.frame,
            bundle.target_col,
            bundle.sensitive_col,
            leakage_columns=bundle.leakage_columns,
            enable_feature_interactions=cfg.data.get("enable_feature_interactions", False),
            max_interactions=cfg.data.get("max_feature_interactions", 0),
        )
        cwr = max((len(y) - y.sum()), 1) / max(y.sum(), 1)
        for seed in seeds:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold, (tr, te) in enumerate(cv.split(X, y)):
                X_train, y_train = X.iloc[tr], y.iloc[tr]
                X_test, y_test = X.iloc[te], y.iloc[te]
                s_test = sens.iloc[te]
                pre = make_preprocessor(X_train)
                Xt = pre.fit_transform(X_train)
                Xv = pre.transform(X_test)
                if hasattr(Xt, "toarray"):
                    Xt = Xt.toarray()
                if hasattr(Xv, "toarray"):
                    Xv = Xv.toarray()
                if name == "no_calibration":
                    est = _make_no_calibration_estimator(seed=seed, class_weight_ratio=cwr, mp=mp)
                else:
                    est = _make_full_estimator(seed=seed, class_weight_ratio=cwr, mp=mp)
                t0 = time.perf_counter()
                est.fit(Xt, y_train.values)
                fit_t = time.perf_counter() - t0
                t0 = time.perf_counter()
                y_pred = est.predict(Xv)
                y_prob = est.predict_proba(Xv)[:, 1]
                pred_t = time.perf_counter() - t0
                perf = compute_performance(y_test.values, y_pred, y_prob)
                fair = compute_fairness(y_test.values, y_pred, s_test.values)
                cal = compute_calibration(y_test.values, y_prob)
                cal_par = compute_calibration_per_group(y_test.values, y_prob, s_test.values)
                row = {
                    "variant": name,
                    "dataset": ds,
                    "seed": seed,
                    "fold": fold,
                    "fit_time_sec": fit_t,
                    "predict_time_sec": pred_t,
                }
                row.update(perf)
                row.update(fair)
                row.update(cal)
                row.update(cal_par)
                row["joint_score"] = _compute_joint_score(perf=perf, fair=fair, weights=dict(weights))
                rows.append(row)
                print(
                    f"[{name:14s}] {ds:25s} seed={seed} fold={fold} "
                    f"f1={perf['f1_macro']:.4f} mcc={perf['mcc']:.4f} "
                    f"dp={fair['demographic_parity_diff']:.4f} ece={cal['ece']:.4f}",
                    flush=True,
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HIES ablation study.")
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Legacy reduced protocol (3 seeds × 3 folds) instead of the "
        "full main-benchmark protocol (12 seeds × 5 folds).",
    )
    args = parser.parse_args()
    cfg = load_config("configs/default.yaml")
    out_dir = Path("artifacts") / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = ["full", "no_pruning", "no_per_sample", "no_threshold", "stacked", "no_calibration"]
    if args.reduced:
        seeds = list(cfg.seeds)[:3]
        n_splits = 3
    else:
        seeds = list(cfg.seeds)
        n_splits = int(cfg.n_splits)
    print(
        f"Ablation protocol: {len(seeds)} seeds × {n_splits} folds "
        f"({'reduced' if args.reduced else 'full'})",
        flush=True,
    )
    rows: list[dict] = []
    for v in variants:
        rows.extend(evaluate_variant(v, cfg, seeds, n_splits=n_splits))
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "results_raw.csv", index=False)
    summary = (
        raw.groupby(["variant", "dataset"], as_index=False)
        .agg(
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            mcc_mean=("mcc", "mean"),
            dp_diff_mean=("demographic_parity_diff", "mean"),
            eod_diff_mean=("equal_opportunity_diff", "mean"),
            joint_score_mean=("joint_score", "mean"),
            ece_mean=("ece", "mean"),
            brier_mean=("brier_score", "mean"),
            fit_time_mean=("fit_time_sec", "mean"),
            predict_time_mean=("predict_time_sec", "mean"),
        )
        .sort_values(["dataset", "f1_macro_mean"], ascending=[True, False])
    )
    summary.to_csv(out_dir / "results_summary.csv", index=False)
    pivot = summary.pivot(
        index="variant",
        columns="dataset",
        values="f1_macro_mean",
    ).round(4)
    pivot.to_csv(out_dir / "table_ablation_f1.csv")
    pivot_joint = summary.pivot(
        index="variant",
        columns="dataset",
        values="joint_score_mean",
    ).round(4)
    pivot_joint.to_csv(out_dir / "table_ablation_joint.csv")
    print(
        json.dumps(
            {
                "raw_rows": len(raw),
                "summary_rows": len(summary),
                "n_seeds": len(seeds),
                "n_splits": n_splits,
                "protocol": "reduced" if args.reduced else "full",
            }
        )
    )


if __name__ == "__main__":
    main()
