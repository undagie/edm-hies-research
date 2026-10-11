"""Plot label-noise robustness (macro-F1 vs noise rate) at IJIES print width.

Usage: python scripts/generate_robustness_plot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_HERE), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fig_style import DATASET_LABELS, FULL_WIDTH_IN, PNG_DPI, apply_ijies_style

METHOD_LABELS = {
    "hies": "HIES (proposed)",
    "calibrated_xgboost": "Calibrated XGBoost",
    "catboost": "CatBoost",
}

OUT = Path("artifacts") / "phase1_analysis"


def main() -> None:
    apply_ijies_style()
    df = pd.read_csv(OUT / "robustness_label_noise_summary.csv")
    datasets = [d for d in DATASET_LABELS if d in set(df["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(FULL_WIDTH_IN, 2.7), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    colors = {"hies": "tab:red", "calibrated_xgboost": "tab:blue", "catboost": "tab:green"}
    markers = {"hies": "*", "calibrated_xgboost": "o", "catboost": "s"}
    for idx, (ax, ds) in enumerate(zip(axes, datasets, strict=False)):
        sub = df[df["dataset"] == ds]
        for m in ["hies", "calibrated_xgboost", "catboost"]:
            s = sub[sub["method"] == m].sort_values("noise_p")
            if s.empty:
                continue
            ax.errorbar(
                s["noise_p"] * 100,
                s["f1_mean"],
                yerr=s["f1_std"],
                label=METHOD_LABELS.get(m, m),
                marker=markers.get(m, "o"),
                color=colors.get(m, "gray"),
                linewidth=1.3,
                markersize=7,
                capsize=2.5,
            )
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("Label-noise rate (%)")
        if idx == 0:
            ax.set_ylabel("Macro-F1")
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="lower left", frameon=False, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(OUT / "robustness_label_noise.png", dpi=PNG_DPI)
    fig.savefig(OUT / "robustness_label_noise.pdf")
    print(f"Wrote {OUT}/robustness_label_noise.png/.pdf")


if __name__ == "__main__":
    main()
