"""Render print-quality top-15 SHAP bar chart from saved UCI Dropout CSV.

Usage: python scripts/generate_shap_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fig_style import FULL_WIDTH_IN, PNG_DPI, apply_ijies_style

OUT = Path("artifacts") / "phase1_analysis"
SRC = OUT / "shap_top15_uci_dropout_success.csv"


def _clean(name: str) -> str:
    for prefix in ("num__", "cat__", "onehot__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def main() -> None:
    apply_ijies_style()
    df = pd.read_csv(SRC).head(15)
    df["feature"] = df["feature"].map(_clean)
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.6))
    ax.barh(df["feature"], df["mean_abs_shap"], color="#4c78a8", height=0.72)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    for y, v in enumerate(df["mean_abs_shap"]):
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, df["mean_abs_shap"].max() * 1.12)
    fig.tight_layout()
    fig.savefig(OUT / "shap_top15_uci_dropout_success.png", dpi=PNG_DPI)
    fig.savefig(OUT / "shap_top15_uci_dropout_success.pdf")
    print(f"Wrote {OUT}/shap_top15_uci_dropout_success.png/.pdf")


if __name__ == "__main__":
    main()
