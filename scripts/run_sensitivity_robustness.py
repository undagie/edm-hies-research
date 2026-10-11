"""Weight-J sensitivity + label-noise robustness (3×3 budget, not full 12×5).

Writes under ``artifacts/phase1_analysis/``.

Usage: python scripts/run_sensitivity_robustness.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
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
from src.evaluation import compute_fairness, compute_performance
from src.models.methods import build_method, make_preprocessor
from src.runner.experiment import _compute_joint_score, _prepare_xy
from src.utils import silence_known_warnings

silence_known_warnings()
warnings.filterwarnings("ignore")

OUT = Path("artifacts") / "phase1_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def joint_with_weights(perf, fair, weights):
    return _compute_joint_score(perf=perf, fair=fair, weights=weights)


def sensitivity_J(cfg, seeds, n_splits, tag: str = "") -> pd.DataFrame:
    base_w = dict(cfg.optimization["objective_weights"])
    perturbations = {
        "default": base_w,
    }
    for k in base_w:
        for sign in (+0.5, -0.5):
            new_w = dict(base_w)
            new_w[k] = max(0.0, base_w[k] * (1.0 + sign))
            label = f"{k}_{'+50%' if sign>0 else '-50%'}"
            perturbations[label] = new_w
    method_params = cfg.optimization["method_params"]
    rows = []
    for ds in cfg.data["dataset_order"]:
        bundle = load_dataset_bundle(ds, cfg.data["raw_dir"], seed=cfg.seed, allow_synthetic=False)
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
                Xt, yt = X.iloc[tr], y.iloc[tr]
                Xv, yv = X.iloc[te], y.iloc[te]
                s_test = sens.iloc[te]
                pre = make_preprocessor(Xt)
                Xt_t = pre.fit_transform(Xt)
                Xv_t = pre.transform(Xv)
                if hasattr(Xt_t, "toarray"):
                    Xt_t = Xt_t.toarray()
                if hasattr(Xv_t, "toarray"):
                    Xv_t = Xv_t.toarray()
                preds = {}
                for m in ["xgboost", "calibrated_xgboost", "catboost", "hies"]:
                    spec = build_method(m, seed=seed, class_weight_ratio=cwr, method_params=method_params)
                    spec.estimator.fit(Xt_t, yt.values)
                    yp = spec.estimator.predict(Xv_t)
                    yprob = spec.estimator.predict_proba(Xv_t)[:, 1]
                    perf = compute_performance(yv.values, yp, yprob)
                    fair = compute_fairness(yv.values, yp, s_test.values)
                    preds[m] = (perf, fair)
                for label, w in perturbations.items():
                    for m, (perf, fair) in preds.items():
                        rows.append(
                            {
                                "dataset": ds,
                                "seed": seed,
                                "fold": fold,
                                "weight_setting": label,
                                "method": m,
                                "joint_score": joint_with_weights(perf, fair, w),
                                "f1_macro": perf["f1_macro"],
                                "mcc": perf["mcc"],
                                "dp_diff": fair["demographic_parity_diff"],
                            }
                        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"sensitivity_J_raw{tag}.csv", index=False)
    summary = (
        df.groupby(["dataset", "weight_setting", "method"], as_index=False)["joint_score"]
        .mean()
        .pivot(index=["dataset", "weight_setting"], columns="method", values="joint_score")
        .reset_index()
    )
    if "hies" in summary.columns:
        nonh = [c for c in summary.columns if c not in ("dataset", "weight_setting", "hies")]
        summary["hies_minus_best_other"] = summary["hies"] - summary[nonh].max(axis=1)
        summary["hies_wins"] = summary["hies_minus_best_other"] > 0
    summary.to_csv(OUT / f"sensitivity_J_summary{tag}.csv", index=False)
    return summary


def robustness_label_noise(cfg, seeds, n_splits, noise_rates=(0.0, 0.05, 0.10, 0.20), tag: str = "") -> pd.DataFrame:
    method_params = cfg.optimization["method_params"]
    rng_master = np.random.default_rng(cfg.seed)
    rows = []
    for ds in cfg.data["dataset_order"]:
        bundle = load_dataset_bundle(ds, cfg.data["raw_dir"], seed=cfg.seed, allow_synthetic=False)
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
                Xt, yt = X.iloc[tr], y.iloc[tr]
                Xv, yv = X.iloc[te], y.iloc[te]
                s_test = sens.iloc[te]
                pre = make_preprocessor(Xt)
                Xt_t = pre.fit_transform(Xt)
                Xv_t = pre.transform(Xv)
                if hasattr(Xt_t, "toarray"):
                    Xt_t = Xt_t.toarray()
                if hasattr(Xv_t, "toarray"):
                    Xv_t = Xv_t.toarray()
                for noise_p in noise_rates:
                    yt_corr = yt.values.copy()
                    if noise_p > 0:
                        rng = np.random.default_rng(seed * 1000 + fold + int(noise_p * 100))
                        flip_idx = rng.choice(
                            np.arange(len(yt_corr)),
                            size=int(round(noise_p * len(yt_corr))),
                            replace=False,
                        )
                        yt_corr[flip_idx] = 1 - yt_corr[flip_idx]
                    for m in ["calibrated_xgboost", "catboost", "hies"]:
                        spec = build_method(m, seed=seed, class_weight_ratio=cwr, method_params=method_params)
                        spec.estimator.fit(Xt_t, yt_corr)
                        yp = spec.estimator.predict(Xv_t)
                        yprob = spec.estimator.predict_proba(Xv_t)[:, 1]
                        perf = compute_performance(yv.values, yp, yprob)
                        rows.append(
                            {
                                "dataset": ds,
                                "seed": seed,
                                "fold": fold,
                                "noise_p": noise_p,
                                "method": m,
                                "f1_macro": perf["f1_macro"],
                                "mcc": perf["mcc"],
                            }
                        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"robustness_label_noise_raw{tag}.csv", index=False)
    summary = (
        df.groupby(["dataset", "noise_p", "method"], as_index=False)
        .agg(f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"))
        .sort_values(["dataset", "noise_p", "method"])
    )
    summary.to_csv(OUT / f"robustness_label_noise_summary{tag}.csv", index=False)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weight-J sensitivity + label-noise robustness.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--only", choices=["both", "sensitivity", "noise"], default="both")
    p.add_argument("--seeds", type=int, default=3, help="Number of seeds from cfg.seeds (default 3, reduced protocol).")
    p.add_argument("--splits", type=int, default=3, help="StratifiedKFold splits (default 3, reduced protocol).")
    p.add_argument("--full", action="store_true", help="Full protocol: all cfg.seeds and 5 splits (overrides --seeds/--splits).")
    p.add_argument("--tag", default="", help="Suffix appended to output CSV filenames (default '': canonical files).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.full:
        seeds = list(cfg.seeds)
        n_splits = int(cfg.n_splits) if getattr(cfg, "n_splits", None) else 5
    else:
        seeds = list(cfg.seeds)[: args.seeds]
        n_splits = args.splits
    print(f"[config] seeds={len(seeds)} splits={n_splits} only={args.only} tag='{args.tag}' "
          f"-> {len(seeds) * n_splits} paired observations per cell")
    if args.only in ("both", "sensitivity"):
        print("[sensitivity-J] running...")
        sJ = sensitivity_J(cfg, seeds, n_splits=n_splits, tag=args.tag)
        print(f"  rows={len(sJ)}; wrote sensitivity_J_summary{args.tag}.csv")
    if args.only in ("both", "noise"):
        print("[robustness] running...")
        rN = robustness_label_noise(cfg, seeds, n_splits=n_splits, tag=args.tag)
        print(f"  rows={len(rN)}; wrote robustness_label_noise_summary{args.tag}.csv")


if __name__ == "__main__":
    main()
