"""Phase-1 analyses on ``artifacts/results/results_raw.csv``.

Wilcoxon/TOST significance, utility, cost, subgroup, calibration tables;
paired-diff boxplot and F1–DP Pareto plots under ``artifacts/phase1_analysis/``.

Usage: python scripts/analyze_phase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "artifacts" / "results" / "results_raw.csv"
OUT = ROOT / "artifacts" / "phase1_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def cliffs_delta_paired(diff: np.ndarray) -> float:
    """Cliff's delta on paired differences: P(diff > 0) - P(diff < 0)."""
    diff = np.asarray(diff, dtype=float)
    return float((diff > 0).mean() - (diff < 0).mean())


def holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * p[idx], 1.0)
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def tost_non_inferiority(
    a: np.ndarray, b: np.ndarray, margin: float = 0.005
) -> dict:
    """TOST non-inferiority of A vs B; H0: mean(A-B) <= -margin; p_ni < 0.05 rejects."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    if n < 2:
        return {"n": int(n), "mean_diff": float(diff.mean()), "p_ni": 1.0, "non_inferior": False}
    mean_d = float(diff.mean())
    sd = float(diff.std(ddof=1))
    if sd == 0:
        p_ni = 0.0 if mean_d > -margin else 1.0
    else:
        se = sd / np.sqrt(n)
        t_stat = (mean_d + margin) / se
        p_ni = float(1.0 - student_t.cdf(t_stat, df=n - 1))
    return {
        "n": int(n),
        "mean_diff": mean_d,
        "p_ni": p_ni,
        "non_inferior": bool(p_ni < 0.05),
    }


def aligned_pairs(raw: pd.DataFrame, dataset: str, m_a: str, m_b: str, metric: str = "f1_macro"):
    a = raw[(raw["dataset"] == dataset) & (raw["method"] == m_a)].sort_values(["seed", "fold"])[metric].to_numpy()
    b = raw[(raw["dataset"] == dataset) & (raw["method"] == m_b)].sort_values(["seed", "fold"])[metric].to_numpy()
    n = min(len(a), len(b))
    return a[:n], b[:n]


def make_significance_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    competitors = [m for m in raw["method"].unique() if m != "hies"]
    for ds in sorted(raw["dataset"].unique()):
        ds_rows = []
        for comp in competitors:
            a, b = aligned_pairs(raw, ds, "hies", comp, "f1_macro")
            if len(a) == 0:
                continue
            diff = a - b
            if not np.allclose(diff, 0):
                p_two = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
            else:
                p_two = 1.0
            tost = tost_non_inferiority(a, b, margin=0.005)
            if len(diff) > 1 and float(diff.std(ddof=1)) > 0:
                res = bootstrap((diff,), np.mean, n_resamples=1000, random_state=0, method="basic")
                ci_low = float(res.confidence_interval.low)
                ci_high = float(res.confidence_interval.high)
            else:
                ci_low = ci_high = float(diff.mean())
            ds_rows.append(
                {
                    "dataset": ds,
                    "comparator": comp,
                    "hies_mean": float(a.mean()),
                    "comp_mean": float(b.mean()),
                    "delta_mean": float(diff.mean()),
                    "delta_ci95_low": ci_low,
                    "delta_ci95_high": ci_high,
                    "cliffs_delta_paired": cliffs_delta_paired(diff),
                    "wilcoxon_p_two_sided": p_two,
                    "tost_margin_005_p_ni": tost["p_ni"],
                    "tost_non_inferior": tost["non_inferior"],
                    "n_pairs": int(len(a)),
                }
            )
        if ds_rows:
            holm = holm_correction(np.array([r["wilcoxon_p_two_sided"] for r in ds_rows]))
            for r, p_h in zip(ds_rows, holm, strict=True):
                r["wilcoxon_p_holm"] = float(p_h)
            rows.extend(ds_rows)
    return pd.DataFrame(rows).sort_values(["dataset", "comparator"])


def _group_class_shares() -> dict[str, dict[str, dict[str, float]]]:
    """Positive/negative class shares per sensitive group (for overall TPR/FPR)."""
    from src.config import load_config
    from src.datasets import load_dataset_bundle
    from src.runner.experiment import _prepare_xy

    cfg = load_config("configs/default.yaml")
    shares: dict[str, dict[str, dict[str, float]]] = {}
    for ds in cfg.data["dataset_order"]:
        bundle = load_dataset_bundle(ds, cfg.data["raw_dir"], seed=cfg.seed, allow_synthetic=False)
        _, y, sens = _prepare_xy(
            bundle.frame,
            bundle.target_col,
            bundle.sensitive_col,
            leakage_columns=bundle.leakage_columns,
            enable_feature_interactions=False,
            max_interactions=0,
        )
        pos = y == 1
        neg = ~pos
        groups = sens.astype(str)
        shares[ds] = {
            "pos": {
                str(g): float((pos & (groups == str(g))).sum()) / int(pos.sum())
                for g in groups.unique()
            },
            "neg": {
                str(g): float((neg & (groups == str(g))).sum()) / int(neg.sum())
                for g in groups.unique()
            },
        }
    return shares


def utility_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Paired TPR/FPR gaps as students per 1,000 (share-weighted overall rates)."""
    shares = _group_class_shares()
    rows = []
    competitors = [m for m in raw["method"].unique() if m != "hies"]
    for ds in sorted(raw["dataset"].unique()):
        ds_shares = shares.get(ds, {})
        groups = [
            c.replace("tpr_grp_", "")
            for c in raw.columns
            if c.startswith("tpr_grp_")
            and raw.loc[raw["dataset"] == ds, c].notna().any()
            and c.replace("tpr_grp_", "") in ds_shares.get("pos", {})
        ]
        if not groups:
            continue
        sub = raw[raw["dataset"] == ds].copy()
        sub["tpr_overall"] = sum(sub[f"tpr_grp_{g}"] * ds_shares["pos"][g] for g in groups)
        sub["fpr_overall"] = sum(sub[f"fpr_grp_{g}"] * ds_shares["neg"][g] for g in groups)
        for comp in competitors:
            a = sub[sub["method"] == "hies"].sort_values(["seed", "fold"])
            b = sub[sub["method"] == comp].sort_values(["seed", "fold"])
            n = min(len(a), len(b))
            if n == 0:
                continue
            d_tpr = a["tpr_overall"].to_numpy()[:n] - b["tpr_overall"].to_numpy()[:n]
            d_fpr = a["fpr_overall"].to_numpy()[:n] - b["fpr_overall"].to_numpy()[:n]
            rows.append(
                {
                    "dataset": ds,
                    "comparator": comp,
                    "hies_tpr_mean": float(a["tpr_overall"].to_numpy()[:n].mean()),
                    "comp_tpr_mean": float(b["tpr_overall"].to_numpy()[:n].mean()),
                    "delta_tpr_mean": float(d_tpr.mean()),
                    "extra_true_positives_per_1000_at_risk": float(d_tpr.mean() * 1000.0),
                    "hies_fpr_mean": float(a["fpr_overall"].to_numpy()[:n].mean()),
                    "comp_fpr_mean": float(b["fpr_overall"].to_numpy()[:n].mean()),
                    "delta_fpr_mean": float(d_fpr.mean()),
                    "false_alarms_avoided_per_1000_not_at_risk": float(-d_fpr.mean() * 1000.0),
                    "n_pairs": int(n),
                }
            )
    out = pd.DataFrame(rows).sort_values(["dataset", "comparator"])
    out.to_csv(OUT / "table_utility.csv", index=False)
    return out


def boxplot_paired_diff(raw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    datasets = sorted(raw["dataset"].unique())
    for ax, ds in zip(axes, datasets, strict=False):
        means = raw[raw["dataset"] == ds].groupby("method")["f1_macro"].mean().sort_values(ascending=False)
        comps = [m for m in means.index if m != "hies"][:4]
        diffs = []
        labels = []
        for comp in comps:
            a, b = aligned_pairs(raw, ds, "hies", comp, "f1_macro")
            if len(a) == 0:
                continue
            diffs.append((a - b))
            labels.append(comp.replace("_", "\n"))
        ax.boxplot(diffs, tick_labels=labels, widths=0.6, showmeans=True)
        ax.axhline(0, linestyle="--", color="gray", alpha=0.6)
        ax.set_title(ds)
        ax.set_ylabel("HIES − comparator (macro-F1)")
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "boxplot_paired_diff.png", dpi=160)
    fig.savefig(OUT / "boxplot_paired_diff.pdf")
    plt.close(fig)


def pareto_plot(raw: pd.DataFrame) -> None:
    summary = (
        raw.groupby(["dataset", "method"], as_index=False)
        .agg(f1=("f1_macro", "mean"), dp=("demographic_parity_diff", "mean"))
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    datasets = sorted(summary["dataset"].unique())
    for ax, ds in zip(axes, datasets, strict=False):
        sub = summary[summary["dataset"] == ds]
        for _, row in sub.iterrows():
            color = "tab:red" if row["method"] == "hies" else "tab:blue"
            marker = "*" if row["method"] == "hies" else "o"
            size = 200 if row["method"] == "hies" else 80
            ax.scatter(row["dp"], row["f1"], s=size, c=color, marker=marker, edgecolors="black", linewidths=0.6)
            ax.annotate(row["method"], (row["dp"], row["f1"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Demographic parity difference (lower is fairer)")
        ax.set_ylabel("Macro-F1 (higher is better)")
        ax.set_title(ds)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "pareto_f1_dp.png", dpi=160)
    fig.savefig(OUT / "pareto_f1_dp.pdf")
    plt.close(fig)


def cost_table(raw: pd.DataFrame) -> pd.DataFrame:
    if "fit_time_sec" not in raw.columns:
        return pd.DataFrame()
    out = (
        raw.groupby(["dataset", "method"], as_index=False)
        .agg(
            fit_mean=("fit_time_sec", "mean"),
            fit_std=("fit_time_sec", "std"),
            predict_mean=("predict_time_sec", "mean"),
        )
        .sort_values(["dataset", "method"])
    )
    out.to_csv(OUT / "table_cost.csv", index=False)
    return out


def subgroup_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-group F1/TPR/FPR/PPV from ``*_grp_*`` columns in raw results."""
    f1_cols = [c for c in raw.columns if c.startswith("f1_macro_grp_")]
    if not f1_cols:
        return pd.DataFrame()
    rows: list[dict] = []
    metric_prefixes = [
        ("f1_macro_grp_", "f1"),
        ("tpr_grp_", "tpr"),
        ("fpr_grp_", "fpr"),
        ("ppv_grp_", "ppv"),
    ]
    for ds in sorted(raw["dataset"].unique()):
        for m in sorted(raw["method"].unique()):
            sub = raw[(raw["dataset"] == ds) & (raw["method"] == m)]
            if sub.empty:
                continue
            entry = {"dataset": ds, "method": m}
            for col in f1_cols:
                grp = col.replace("f1_macro_grp_", "")
                for prefix, short in metric_prefixes:
                    target = f"{prefix}{grp}"
                    if target in sub.columns:
                        vals = sub[target].dropna()
                        if len(vals) > 0:
                            entry[f"{short}_{grp}_mean"] = float(vals.mean())
                            entry[f"{short}_{grp}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            rows.append(entry)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table_subgroup_metrics.csv", index=False)
    # Legacy filename for existing paper refs.
    out.to_csv(OUT / "table_subgroup_f1.csv", index=False)
    return out


def calibration_table(raw: pd.DataFrame) -> pd.DataFrame:
    if "ece" not in raw.columns:
        return pd.DataFrame()
    out = (
        raw.groupby(["dataset", "method"], as_index=False)
        .agg(
            ece_mean=("ece", "mean"),
            brier_mean=("brier_score", "mean"),
            cal_par_mean=("calibration_parity_diff", "mean"),
        )
        .sort_values(["dataset", "ece_mean"])
    )
    out.to_csv(OUT / "table_calibration.csv", index=False)
    return out


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"Missing {RAW}; run main benchmark first.")
    raw = pd.read_csv(RAW)
    print(f"Loaded {len(raw)} rows, methods={sorted(raw['method'].unique())}")

    sig = make_significance_table(raw)
    sig.to_csv(OUT / "table_significance_tost.csv", index=False)
    print(f"Wrote significance/TOST: {len(sig)} comparisons")

    util = utility_table(raw)
    print(f"Wrote utility table: {len(util)} rows")

    cost = cost_table(raw)
    print(f"Wrote cost table: {len(cost)} rows")

    sub = subgroup_table(raw)
    print(f"Wrote subgroup table: {len(sub)} rows")

    cal = calibration_table(raw)
    print(f"Wrote calibration table: {len(cal)} rows")

    boxplot_paired_diff(raw)
    pareto_plot(raw)
    print(f"Plots written to {OUT}/*.png/.pdf")


if __name__ == "__main__":
    main()
