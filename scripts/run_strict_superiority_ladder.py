"""Compare tradeoff target vs baseline across objective-weight profiles (strict superiority style)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from src.config import load_config
from src.runner.experiment import _build_significance_table, finalize_mlflow, run_benchmark


def _extract_stage_metrics(summary: pd.DataFrame, sig: pd.DataFrame, baseline_method: str, target_method: str) -> tuple[list[dict], int]:
    rows: list[dict] = []
    strict_hits = 0
    for dataset, sub in summary.groupby("dataset"):
        by_method = {row["method"]: row for _, row in sub.iterrows()}
        base = by_method.get(baseline_method)
        targ = by_method.get(target_method)
        if base is None or targ is None:
            continue
        delta = float(targ["f1_macro_mean"] - base["f1_macro_mean"])
        sig_row = sig[(sig["dataset"] == dataset) & (sig["method"] == target_method)]
        if sig_row.empty:
            holm = 1.0
            bh = 1.0
        else:
            holm = float(sig_row.iloc[0]["p_value_holm"])
            bh = float(sig_row.iloc[0]["p_value_bh"])
        strict_ok = delta > 0 and holm < 0.05 and bh < 0.05
        strict_hits += int(strict_ok)
        rows.append(
            {
                "dataset": dataset,
                "delta_f1_macro": delta,
                "p_value_holm": holm,
                "p_value_bh": bh,
                "strict_ok": strict_ok,
            }
        )
    return rows, strict_hits


def _run_stage(stage_name: str, cfg, baseline_method: str, target_method: str):
    cfg = deepcopy(cfg)
    cfg.methods = [baseline_method, target_method]
    cfg.outputs = None
    raw, summary = run_benchmark(cfg, run_id=f"strict_{stage_name}")
    finalize_mlflow()
    sig = _build_significance_table(raw)
    metrics_rows, strict_hits = _extract_stage_metrics(summary, sig, baseline_method, target_method)
    for row in metrics_rows:
        row["stage"] = stage_name
    return metrics_rows, strict_hits


def main() -> None:
    cfg = load_config("configs/default.yaml")
    opt = cfg.optimization if isinstance(cfg.optimization, dict) else {}
    baseline_method = str(opt.get("tradeoff_baseline_method", "xgboost"))
    target_method = str(opt.get("tradeoff_target_method", "calibrated_xgboost"))

    profiles = {
        "utility_high": {"f1_macro": 0.86, "mcc": 0.48, "dp_diff": 0.04, "eod_diff": 0.03},
        "balanced_default": {"f1_macro": 0.82, "mcc": 0.45, "dp_diff": 0.06, "eod_diff": 0.05},
        "fairness_heavier": {"f1_macro": 0.76, "mcc": 0.4, "dp_diff": 0.1, "eod_diff": 0.08},
    }

    all_rows: list[dict] = []
    scoreboard: list[dict] = []
    best_stage = None
    best_hits = -1

    for stage_name, weights in profiles.items():
        stage_cfg = deepcopy(cfg)
        stage_cfg.optimization = deepcopy(cfg.optimization)
        stage_cfg.optimization["objective_weights"] = weights
        stage_rows, strict_hits = _run_stage(stage_name, stage_cfg, baseline_method, target_method)
        all_rows.extend(stage_rows)
        scoreboard.append({"stage": stage_name, "strict_hits": strict_hits})
        if strict_hits > best_hits:
            best_hits = strict_hits
            best_stage = stage_name

    out_dir = Path("artifacts/strict_ladder")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_dir / "strict_ladder_results.csv", index=False)
    pd.DataFrame(scoreboard).sort_values("strict_hits", ascending=False).to_csv(
        out_dir / "strict_ladder_scoreboard.csv", index=False
    )

    winner_cfg = load_config("configs/default.yaml")
    winner_cfg.optimization = deepcopy(cfg.optimization)
    winner_cfg.optimization["objective_weights"] = profiles[best_stage or "balanced_default"]

    winner_payload = {
        "seed": winner_cfg.seed,
        "seeds": winner_cfg.seeds,
        "n_splits": winner_cfg.n_splits,
        "data": winner_cfg.data,
        "fairness": winner_cfg.fairness,
        "metrics": winner_cfg.metrics,
        "methods": winner_cfg.methods,
        "optimization": winner_cfg.optimization,
        "outputs": winner_cfg.outputs,
    }
    with (out_dir / "strict_best_config.yaml").open("w", encoding="utf-8") as fp:
        yaml.safe_dump(winner_payload, fp, sort_keys=False)


if __name__ == "__main__":
    main()
