"""Random-search sweep for HIES only (cached folds; skip other methods).

Selection: maximin ``min_gap_to_target``; ``harmonic_mean_f1`` secondary.

Usage: python scripts/sweep_hies_adaptive.py --phase A --n-configs 60 --n-seeds 1 --n-folds 3
"""

from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from src.config import load_config
from src.datasets.loader import load_dataset_bundle
from src.models.methods import build_method, make_preprocessor
from src.runner.experiment import _prepare_xy
from src.utils import silence_known_warnings

silence_known_warnings()


@dataclass
class FoldData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_train: int
    class_ratio: float
    class_weight_ratio: float


def _prepare_dataset_folds(
    dataset_name: str,
    raw_dir: str,
    *,
    seeds: list[int],
    n_folds: int,
    config_seed: int,
    enable_feature_interactions: bool,
    max_interactions: int,
    allow_synthetic: bool,
) -> list[tuple[int, int, FoldData]]:
    bundle = load_dataset_bundle(
        dataset_name,
        raw_dir,
        seed=config_seed,
        allow_synthetic=allow_synthetic,
    )
    X, y, _sensitive = _prepare_xy(
        bundle.frame,
        bundle.target_col,
        bundle.sensitive_col,
        leakage_columns=bundle.leakage_columns,
        enable_feature_interactions=enable_feature_interactions,
        max_interactions=max_interactions,
    )
    out: list[tuple[int, int, FoldData]] = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            X_train = X.iloc[tr].copy()
            y_train = y.iloc[tr].astype(int).to_numpy()
            X_test = X.iloc[te].copy()
            y_test = y.iloc[te].astype(int).to_numpy()
            pre = make_preprocessor(X_train)
            X_train_t = pre.fit_transform(X_train)
            X_test_t = pre.transform(X_test)
            if hasattr(X_train_t, "toarray"):
                X_train_t = X_train_t.toarray()
            if hasattr(X_test_t, "toarray"):
                X_test_t = X_test_t.toarray()
            n_train = int(len(y_train))
            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            class_weight_ratio = float(max(n_neg, 1) / max(n_pos, 1))
            class_ratio = float(max(n_pos, n_neg) / max(min(n_pos, n_neg), 1))
            out.append(
                (
                    seed,
                    fold,
                    FoldData(
                        X_train=X_train_t,
                        y_train=y_train,
                        X_test=X_test_t,
                        y_test=y_test,
                        n_train=n_train,
                        class_ratio=class_ratio,
                        class_weight_ratio=class_weight_ratio,
                    ),
                )
            )
    return out


def _evaluate_config(
    cfg: dict,
    fold_cache: dict[str, list[tuple[int, int, FoldData]]],
) -> dict:
    rows = []
    for dataset_name, folds in fold_cache.items():
        for seed, fold_idx, fd in folds:
            spec = build_method(
                "hies",
                seed=seed,
                class_weight_ratio=fd.class_weight_ratio,
                method_params={"hies": deepcopy(cfg)},
            )
            est = spec.estimator
            est.fit(fd.X_train, fd.y_train)
            y_pred = est.predict(fd.X_test)
            f1m = float(f1_score(fd.y_test, y_pred, average="macro"))
            mcc = float(matthews_corrcoef(fd.y_test, y_pred))
            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "fold": fold_idx,
                    "f1_macro": f1m,
                    "mcc": mcc,
                }
            )
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("dataset", as_index=False)
        .agg(
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            mcc_mean=("mcc", "mean"),
            n=("f1_macro", "count"),
        )
        .sort_values("dataset")
    )
    return {"raw": df, "summary": summary}


def _config_to_record(
    cfg: dict,
    summary: pd.DataFrame,
    targets: dict[str, float],
) -> dict:
    flat = {}
    for d, t in targets.items():
        row = summary[summary["dataset"] == d]
        if row.empty:
            flat[f"{d}_f1"] = float("nan")
            flat[f"{d}_gap_to_target"] = float("-inf")
        else:
            f1 = float(row.iloc[0]["f1_macro_mean"])
            flat[f"{d}_f1"] = f1
            flat[f"{d}_gap_to_target"] = f1 - float(t)
    flat["min_gap_to_target"] = min(flat[f"{d}_gap_to_target"] for d in targets)
    flat["mean_f1"] = float(summary["f1_macro_mean"].mean()) if not summary.empty else float("nan")
    f1_pos = [
        flat[f"{d}_f1"]
        for d in targets
        if np.isfinite(flat[f"{d}_f1"]) and flat[f"{d}_f1"] > 0
    ]
    if len(f1_pos) == len(targets):
        flat["harmonic_mean_f1"] = float(len(f1_pos) / sum(1.0 / x for x in f1_pos))
    else:
        flat["harmonic_mean_f1"] = float("nan")
    flat["config_json"] = json.dumps(cfg, sort_keys=True)
    return flat


def _sample_phase_a_config(rng: random.Random) -> dict:
    auto = {
        "enabled": True,
        "n_large_split": rng.choice([1500, 2000, 2500]),
        "imbalance_split": rng.choice([2.0, 2.5, 3.0]),
        "imbalanced_mcc_boost": rng.choice([0.0, 0.05, 0.10, 0.15]),
        "small": {
            "prune_delta": rng.choice([0.020, 0.030, 0.040, 0.050]),
            "temperature": rng.choice([0.05, 0.07, 0.10, 0.15]),
            "mcc_weight": rng.choice([0.40, 0.50, 0.60, 0.70]),
            "threshold_grid": rng.choice(
                [
                    [0.40, 0.45, 0.50, 0.55],
                    [0.40, 0.43, 0.45, 0.47, 0.50, 0.53, 0.55],
                    [0.42, 0.45, 0.48, 0.50, 0.52, 0.55],
                    [0.43, 0.46, 0.50, 0.54, 0.57],
                ]
            ),
        },
        "large": {
            "prune_delta": rng.choice([0.010, 0.015, 0.020, 0.025, 0.030]),
            "temperature": rng.choice([0.04, 0.06, 0.08, 0.10]),
            "mcc_weight": rng.choice([0.40, 0.50, 0.60, 0.70, 0.80]),
            "threshold_grid": rng.choice(
                [
                    [0.40, 0.45, 0.50, 0.55],
                    [0.40, 0.43, 0.46, 0.50, 0.54],
                    [0.42, 0.45, 0.48, 0.50, 0.52, 0.55],
                    [0.43, 0.46, 0.50, 0.54, 0.57],
                ]
            ),
        },
        # Wide / narrow overlays on post-encoding dimensionality (None = routing off).
        "feature_dim_split": rng.choice([None, None, 65, 80, 100, 120]),
        "wide": {
            "prune_delta": rng.choice([0.035, 0.045, 0.055]),
            "temperature": rng.choice([0.08, 0.10, 0.12]),
        },
        "narrow": {
            "prune_delta": rng.choice([0.015, 0.022, 0.028]),
            "temperature": rng.choice([0.05, 0.06, 0.07]),
        },
    }
    return {
        "min_keep": 3,
        "margin_floor": rng.choice([0.03, 0.05, 0.08, 0.10]),
        "cv_splits": 3,
        "calibration_cv": 2,
        "calibration_method": rng.choice(["auto", "sigmoid"]),
        "n_train_threshold": rng.choice([1500, 2000, 2500, 4000]),
        "tune_threshold": True,
        "threshold_clip": [0.30, 0.70],
        "objective_weights": {
            "f1_macro": 0.88,
            "mcc": 0.50,
            "dp_diff": 0.03,
            "eod_diff": 0.02,
        },
        "auto_config": auto,
        "xgb": {"n_estimators": 240, "max_depth": 4, "learning_rate": 0.05},
        "rf": {"n_estimators": 400},
        "hgb": {"max_iter": 260, "max_depth": 6},
        "lgbm": {"n_estimators": 260, "num_leaves": 47, "learning_rate": 0.05},
    }


def _phase_b_promote_config(cfg: dict) -> dict:
    full = deepcopy(cfg)
    full["cv_splits"] = 5
    full["calibration_cv"] = 3
    return full


def _load_top_configs(top_from: Path, n_configs: int) -> list[dict]:
    df = pd.read_csv(top_from)
    df = df.sort_values("min_gap_to_target", ascending=False).head(n_configs)
    return [json.loads(s) for s in df["config_json"].tolist()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HIES random-search sweep.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--phase", choices=["A", "B"], required=True)
    p.add_argument("--n-configs", type=int, default=60)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--top-from", type=str, default=None, help="CSV produced by phase A.")
    p.add_argument("--out-dir", type=str, default="artifacts/hies_adaptive_sweep")
    p.add_argument("--rng-seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds_master = list(cfg.seeds)
    seeds_used = seeds_master[: max(1, args.n_seeds)]
    n_folds = int(args.n_folds)
    print(f"[sweep] phase={args.phase} configs={args.n_configs} seeds={seeds_used} folds={n_folds}")
    print("[sweep] caching dataset folds...")
    fold_cache: dict[str, list[tuple[int, int, FoldData]]] = {}
    for dataset_name in cfg.data["dataset_order"]:
        t0 = time.time()
        fold_cache[dataset_name] = _prepare_dataset_folds(
            dataset_name,
            cfg.data["raw_dir"],
            seeds=seeds_used,
            n_folds=n_folds,
            config_seed=int(cfg.seed),
            enable_feature_interactions=bool(cfg.data.get("enable_feature_interactions", False)),
            max_interactions=int(cfg.data.get("max_feature_interactions", 0)),
            allow_synthetic=bool(cfg.data.get("use_synthetic_if_missing", False)),
        )
        print(f"[sweep] cached {dataset_name}: {len(fold_cache[dataset_name])} folds in {time.time()-t0:.1f}s")

    targets = {
        "uci_dropout_success": 0.8540,
        "uci_student": 0.9128,
        "xapi": 0.9114,
    }

    rng = random.Random(int(args.rng_seed))
    if args.phase == "A":
        configs = [_sample_phase_a_config(rng) for _ in range(int(args.n_configs))]
    else:
        if not args.top_from:
            raise SystemExit("Phase B requires --top-from <phase A CSV>.")
        candidates = _load_top_configs(Path(args.top_from), int(args.n_configs))
        configs = [_phase_b_promote_config(c) for c in candidates]
        print(f"[sweep] promoted {len(configs)} configs to phase B (cv_splits=5, calibration_cv=3)")

    results: list[dict] = []
    raw_records: list[pd.DataFrame] = []
    for idx, cfg_one in enumerate(configs):
        t0 = time.time()
        try:
            res = _evaluate_config(cfg_one, fold_cache)
        except Exception as exc:  # noqa: BLE001
            print(f"[sweep] config {idx} failed: {exc}")
            continue
        rec = _config_to_record(cfg_one, res["summary"], targets)
        rec["config_id"] = idx
        rec["elapsed_sec"] = round(time.time() - t0, 1)
        results.append(rec)
        raw = res["raw"].copy()
        raw["config_id"] = idx
        raw_records.append(raw)
        hm = rec.get("harmonic_mean_f1", float("nan"))
        print(
            f"[sweep] cfg {idx + 1}/{len(configs)} | t={rec['elapsed_sec']:.1f}s | "
            f"min_gap={rec['min_gap_to_target']:+.4f} | hm_f1={hm:.4f} | mean_f1={rec['mean_f1']:.4f} | "
            f"dropout={rec['uci_dropout_success_f1']:.4f} student={rec['uci_student_f1']:.4f} xapi={rec['xapi_f1']:.4f}"
        )

    if not results:
        raise SystemExit("[sweep] no results collected.")
    df = pd.DataFrame(results)
    sort_cols = ["min_gap_to_target", "harmonic_mean_f1"]
    df = df.sort_values(sort_cols, ascending=[False, False], na_position="last")
    raw_all = pd.concat(raw_records, ignore_index=True)
    out_csv = out_dir / f"results_phase{args.phase}.csv"
    raw_csv = out_dir / f"raw_phase{args.phase}.csv"
    df.to_csv(out_csv, index=False)
    raw_all.to_csv(raw_csv, index=False)
    print(f"[sweep] wrote {out_csv} ({len(df)} rows)")
    print(f"[sweep] wrote {raw_csv} ({len(raw_all)} rows)")

    best = df.iloc[0]
    best_cfg = json.loads(best["config_json"])
    best_path = out_dir / f"best_phase{args.phase}.yaml"
    best_path.write_text(yaml.safe_dump({"hies": best_cfg}, sort_keys=False), encoding="utf-8")
    print(f"[sweep] best config (min_gap_to_target={best['min_gap_to_target']:+.4f}) written to {best_path}")


if __name__ == "__main__":
    main()
