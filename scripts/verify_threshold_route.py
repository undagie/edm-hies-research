"""Verify main vs ablation HIES differ only by threshold (same fitted probs).

Fits ablation ``full`` HIES; scores at internal ``threshold_`` and at the
runner ``decision_threshold``. Matches imply the Table 4/7 F1 gap is
threshold-only. Writes ``artifacts/ablation/verify_threshold_route.csv``.

Usage: python scripts/verify_threshold_route.py [--datasets ...] [--seeds N]
"""

from __future__ import annotations

import argparse
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
from src.evaluation import compute_performance
from src.models.methods import build_method, make_preprocessor
from src.runner.experiment import _prepare_xy
from src.utils import silence_known_warnings

silence_known_warnings()

OUT = Path("artifacts") / "ablation" / "verify_threshold_route.csv"
MAIN_RAW = Path("artifacts") / "results" / "results_raw.csv"
ABL_RAW = Path("artifacts") / "ablation" / "results_raw.csv"

# Cheapest datasets first so partial runs yield early evidence.
DATASET_ORDER = ["uci_student", "xapi", "uci_dropout_success"]

COLUMNS = [
    "dataset",
    "seed",
    "fold",
    "threshold_internal",
    "threshold_runner",
    "f1_internal",
    "f1_runner",
    "f1_ablation_full",
    "f1_main_hies",
    "match_ablation",
    "match_main",
    "fit_time_sec",
]


def _make_full_estimator(seed: int, class_weight_ratio: float, mp: dict):
    spec = build_method(
        "hies",
        seed=seed,
        class_weight_ratio=class_weight_ratio,
        method_params={"hies": deepcopy(mp)},
    )
    return spec.estimator


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the HIES threshold route.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of dataset keys.")
    parser.add_argument("--seeds", type=int, default=0, help="Use only the first N seeds (0 = all).")
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml")
    main_raw = pd.read_csv(MAIN_RAW)
    main_hies = main_raw[main_raw["method"] == "hies"].set_index(["dataset", "seed", "fold"])
    abl_raw = pd.read_csv(ABL_RAW)
    abl_full = abl_raw[abl_raw["variant"] == "full"].set_index(["dataset", "seed", "fold"])

    seeds = list(cfg.seeds)
    if args.seeds:
        seeds = seeds[: args.seeds]
    n_splits = int(cfg.n_splits)
    datasets = args.datasets or [d for d in DATASET_ORDER if d in cfg.data["dataset_order"]]

    # Match scripts/run_ablation.py ``full`` construction.
    base_mp = deepcopy(cfg.optimization["method_params"]["hies"])
    base_mp["tune_threshold"] = True

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")

    rows: list[dict] = []
    for ds in datasets:
        bundle = load_dataset_bundle(
            ds, cfg.data["raw_dir"], seed=cfg.seed, allow_synthetic=False
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
                pre = make_preprocessor(X_train)
                Xt = pre.fit_transform(X_train)
                Xv = pre.transform(X_test)
                if hasattr(Xt, "toarray"):
                    Xt = Xt.toarray()
                if hasattr(Xv, "toarray"):
                    Xv = Xv.toarray()
                est = _make_full_estimator(seed=seed, class_weight_ratio=cwr, mp=base_mp)
                t0 = time.perf_counter()
                est.fit(Xt, y_train.values)
                fit_t = time.perf_counter() - t0
                y_prob = est.predict_proba(Xv)[:, 1]

                key = (ds, seed, fold)
                thr_int = float(est.threshold_)
                thr_run = float(main_hies.loc[key, "decision_threshold"])
                f1_int = compute_performance(
                    y_test.values, (y_prob >= thr_int).astype(int), y_prob
                )["f1_macro"]
                f1_run = compute_performance(
                    y_test.values, (y_prob >= thr_run).astype(int), y_prob
                )["f1_macro"]
                f1_abl = float(abl_full.loc[key, "f1_macro"])
                f1_main = float(main_hies.loc[key, "f1_macro"])

                row = {
                    "dataset": ds,
                    "seed": seed,
                    "fold": fold,
                    "threshold_internal": thr_int,
                    "threshold_runner": thr_run,
                    "f1_internal": f1_int,
                    "f1_runner": f1_run,
                    "f1_ablation_full": f1_abl,
                    "f1_main_hies": f1_main,
                    "match_ablation": bool(np.isclose(f1_int, f1_abl, rtol=0, atol=1e-12)),
                    "match_main": bool(np.isclose(f1_run, f1_main, rtol=0, atol=1e-12)),
                    "fit_time_sec": fit_t,
                }
                rows.append(row)
                with OUT.open("a", encoding="utf-8") as fh:
                    fh.write(",".join(str(row[c]) for c in COLUMNS) + "\n")
                print(
                    f"{ds:22s} seed={seed:3d} fold={fold} "
                    f"thr_int={thr_int:.2f} thr_run={thr_run:.2f} "
                    f"abl={f1_int:.6f}/{f1_abl:.6f} {'OK' if row['match_ablation'] else 'MISMATCH'} "
                    f"main={f1_run:.6f}/{f1_main:.6f} {'OK' if row['match_main'] else 'MISMATCH'}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    print("\n=== summary ===", flush=True)
    for ds, g in df.groupby("dataset"):
        print(
            f"{ds:22s} n={len(g):3d} "
            f"reproduces ablation full: {int(g.match_ablation.sum())}/{len(g)} "
            f"reproduces main HIES: {int(g.match_main.sum())}/{len(g)} "
            f"mean f1 internal={g.f1_internal.mean():.4f} runner={g.f1_runner.mean():.4f} "
            f"(ablation {g.f1_ablation_full.mean():.4f}, main {g.f1_main_hies.mean():.4f})",
            flush=True,
        )
    print(
        f"TOTAL n={len(df)} ablation {int(df.match_ablation.sum())}/{len(df)} "
        f"main {int(df.match_main.sum())}/{len(df)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
