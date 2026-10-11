"""Pick the best-performing method per dataset from a benchmark summary table."""

from __future__ import annotations

import pandas as pd


def select_best_per_dataset(
    summary: pd.DataFrame,
    primary_metric: str = "f1_macro_mean",
    tiebreakers: list[str] | None = None,
    exclude_methods: tuple[str, ...] = ("dummy_most_frequent_baseline",),
) -> pd.DataFrame:
    """Top method per dataset by primary metric, then tiebreakers (dummy excluded)."""
    tb = tiebreakers or ["joint_score_mean", "mcc_mean", "pr_auc_mean"]
    req = {"dataset", "method", primary_metric}
    missing = req - set(summary.columns)
    if missing:
        raise ValueError(f"summary missing columns: {sorted(missing)}")

    sort_cols = [primary_metric] + [c for c in tb if c in summary.columns]
    ascending = [False] * len(sort_cols)

    rows: list[pd.Series] = []
    for _, group in summary.groupby("dataset", sort=True):
        g = group.copy()
        if exclude_methods:
            g = g[~g["method"].isin(exclude_methods)]
        if g.empty:
            continue
        best = g.sort_values(sort_cols, ascending=ascending).iloc[0]
        rows.append(best)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)
