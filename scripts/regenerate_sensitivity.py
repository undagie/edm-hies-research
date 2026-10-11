"""Regenerate ``sensitivity_J_summary.csv`` from full 12×5 main results.

Uses saved per-(seed,fold) metrics so numbers match Table 3 (not the 3×3
sensitivity run). Competitors: CatBoost, XGBoost (Cal-XGB excluded as a HIES base).

Usage: python scripts/regenerate_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path("artifacts/results/results_raw.csv")
OUT_PATH = Path("artifacts/phase1_analysis/sensitivity_J_summary.csv")

BASE_WEIGHTS = {"f1_macro": 0.88, "mcc": 0.50, "dp_diff": 0.03, "eod_diff": 0.02}
COMPETITORS = ("catboost", "xgboost")
DATASETS = ("uci_dropout_success", "uci_student", "xapi")


def make_perturbations() -> dict[str, dict[str, float]]:
    out = {"default": dict(BASE_WEIGHTS)}
    for k, base in BASE_WEIGHTS.items():
        for sign in (+0.5, -0.5):
            label = f"{k}_{'+50%' if sign > 0 else '-50%'}"
            w = dict(BASE_WEIGHTS)
            w[k] = max(0.0, base * (1.0 + sign))
            out[label] = w
    return out


def joint(df: pd.DataFrame, w: dict[str, float]) -> float:
    return (
        w["f1_macro"] * df["f1_macro"]
        + w["mcc"] * df["mcc"]
        - w["dp_diff"] * df["demographic_parity_diff"]
        - w["eod_diff"] * df["equal_opportunity_diff"]
    ).mean()


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    sub = raw[raw["method"].isin(("hies",) + COMPETITORS)].copy()
    rows = []
    for ds in DATASETS:
        ds_df = sub[sub["dataset"] == ds]
        for label, w in make_perturbations().items():
            scores = {m: joint(ds_df[ds_df["method"] == m], w) for m in ("hies",) + COMPETITORS}
            best_other = max(scores[m] for m in COMPETITORS)
            rows.append(
                {
                    "dataset": ds,
                    "weight_setting": label,
                    **scores,
                    "hies_minus_best_other": scores["hies"] - best_other,
                    "hies_wins": bool(scores["hies"] - best_other > 0),
                }
            )
    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(out.to_string(index=False))
    print()
    print("Wins per dataset:")
    print(out.groupby("dataset")["hies_wins"].sum().to_string())


if __name__ == "__main__":
    main()
