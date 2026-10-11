"""Generate Figures 3–6 from existing artifact tables into ``artifacts/diagrams/``.

Usage: python scripts/generate_paper_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import silence_known_warnings

silence_known_warnings()


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    diagrams_dir = base / "artifacts" / "diagrams"
    diagrams_dir.mkdir(parents=True, exist_ok=True)

    main_table = pd.read_csv(base / "artifacts/tables/table_main_results.csv")
    tradeoff_table = pd.read_csv(base / "artifacts/tables/table_tradeoff.csv")
    sensitivity_table = pd.read_csv(base / "artifacts/tables/table_robustness_sensitivity.csv")
    seed_tradeoff_table = pd.read_csv(base / "artifacts/tables/table_robustness_seed_tradeoff.csv")

    if not tradeoff_table.empty:
        baseline_m = str(tradeoff_table.iloc[0]["baseline_method"])
        target_m = str(tradeoff_table.iloc[0]["target_method"])
    else:
        baseline_m = "xgboost"
        target_m = "hies"

    _plot_macro_f1_ci(main_table, diagrams_dir / "figure3_macro_f1_ci.png", baseline_m, target_m)
    _plot_fairness_utility_map(main_table, tradeoff_table, diagrams_dir / "figure4_fairness_utility_pareto.png", target_m)
    _plot_sensitivity_profile(sensitivity_table, diagrams_dir / "figure5_robustness_profiles.png")
    _plot_seed_robustness(seed_tradeoff_table, diagrams_dir / "figure6_seed_robustness.png", baseline_m, target_m)

    print("Generated Figure 3-6 in artifacts/diagrams")


def _plot_macro_f1_ci(
    main_table: pd.DataFrame, output_path: Path, baseline_method: str, target_method: str
) -> None:
    subset = main_table[main_table["method"].isin([baseline_method, target_method])].copy()
    datasets = ["uci_student", "uci_dropout_success", "xapi"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(datasets))
    width = 0.34

    for idx, method in enumerate([baseline_method, target_method]):
        section = subset[subset["method"] == method].set_index("dataset").loc[datasets]
        y = section["f1_macro_mean"].to_numpy()
        yerr = [
            y - section["f1_macro_ci_low"].to_numpy(),
            section["f1_macro_ci_high"].to_numpy() - y,
        ]
        ax.bar(x + (idx - 0.5) * width, y, width=width, label=method, alpha=0.85)
        ax.errorbar(
            x + (idx - 0.5) * width,
            y,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 with 95% CI: target vs baseline")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_fairness_utility_map(
    main_table: pd.DataFrame, tradeoff_table: pd.DataFrame, output_path: Path, target_method: str
) -> None:
    section = main_table[["dataset", "method", "f1_macro_mean", "dp_diff_mean", "eod_diff_mean"]].copy()
    section["fairness_abs"] = section["dp_diff_mean"].abs() + section["eod_diff_mean"].abs()

    highlight = (
        tradeoff_table.set_index("dataset")[["fairness_win_rate", "joint_win_rate"]]
        if not tradeoff_table.empty
        else pd.DataFrame()
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for dataset, group in section.groupby("dataset"):
        ax.scatter(group["fairness_abs"], group["f1_macro_mean"], label=dataset, s=48, alpha=0.8)
        tgt_rows = group[group["method"] == target_method]
        if tgt_rows.empty:
            continue
        tgt_point = tgt_rows.iloc[0]
        if not highlight.empty and dataset in highlight.index:
            annotation = (
                f"{dataset}: {target_method} "
                f"(fw={highlight.loc[dataset, 'fairness_win_rate']:.2f}, "
                f"jw={highlight.loc[dataset, 'joint_win_rate']:.2f})"
            )
        else:
            annotation = f"{dataset}: {target_method}"
        ax.annotate(
            annotation,
            (tgt_point["fairness_abs"], tgt_point["f1_macro_mean"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("|DP-diff| + |EOD-diff| (lower is fairer)")
    ax.set_ylabel("Macro-F1 (higher is better)")
    ax.set_title("Utility-Fairness Trade-off Map Across Methods")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_sensitivity_profile(sensitivity_table: pd.DataFrame, output_path: Path) -> None:
    profile_order = ["utility_high", "balanced_default", "fairness_heavier"]
    delta_col = "delta_f1" if "delta_f1" in sensitivity_table.columns else "delta_f1_macro_mean"
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for dataset, group in sensitivity_table.groupby("dataset"):
        section = group.set_index("profile").loc[profile_order]
        ax.plot(section.index, section[delta_col], marker="o", label=dataset)

    ax.axhline(0, color="gray", linewidth=1)
    ax.set_ylabel("Delta Macro-F1 (target - baseline)")
    ax.set_title("Objective-Weight Sensitivity of Delta Macro-F1")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_seed_robustness(
    seed_tradeoff_table: pd.DataFrame, output_path: Path, baseline_method: str, target_method: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        seed_tradeoff_table["dataset"],
        seed_tradeoff_table["delta_f1_macro_mean"],
        yerr=seed_tradeoff_table["delta_f1_macro_std"],
        capsize=5,
        alpha=0.85,
    )
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_ylabel("Delta Macro-F1 mean ± std (paired folds)")
    ax.set_title(f"Extended-seed robustness: {target_method} vs {baseline_method}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
