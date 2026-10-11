"""Generate HIES architecture diagram (PNG + SVG + PDF) at IJIES print width.

Usage: python scripts/generate_hies_architecture.py
Output: artifacts/diagrams/hies_architecture.[png|svg|pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fig_style import FULL_WIDTH_IN, PNG_DPI, apply_ijies_style

OUT = Path("artifacts") / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, xy, w, h, text, color="#cce5ff", edge="#003366", fontsize=10):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=1.0, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, src, dst, color="#333", style="-|>"):
    ax.annotate(
        "",
        xy=dst,
        xytext=src,
        arrowprops=dict(
            arrowstyle=style, color=color, linewidth=1.3,
            mutation_scale=16, shrinkA=0, shrinkB=0,
        ),
    )


def main() -> None:
    apply_ijies_style()
    # Full-bleed axes + matched aspect: default margins crop width in Word.
    xlo, xhi, ylo, yhi = -0.15, 18.1, 0.55, 8.95
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, FULL_WIDTH_IN * (yhi - ylo) / (xhi - xlo)))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.axis("off")
    flow_y = 4.75

    box(ax, (0.15, flow_y - 0.85), 1.6, 1.7, "Training\nfold\n$(X, y)$", color="#fff2cc")
    box(ax, (2.4, flow_y - 0.85), 1.7, 1.7, "Preprocess\n(impute +\none-hot)", color="#fff2cc")

    bh = 1.3
    base_c = [8.05, 6.4, 4.75, 3.1, 1.45]
    base_y = [c - bh / 2 for c in base_c]
    base_labels = [
        "XGBoost\n+ sigmoid cal.",
        "Random Forest\n+ sigmoid cal.",
        "HistGB\n+ sigmoid cal.",
        "LightGBM\n+ sigmoid cal.",
        "CatBoost\n+ sigmoid cal.",
    ]
    bx, bw = 5.4, 2.7
    for y, lbl in zip(base_y, base_labels, strict=False):
        box(ax, (bx, y), bw, bh, lbl, color="#cce5ff")

    ox, ow = 9.3, 2.2
    box(ax, (ox, flow_y - 0.85), ow, 1.7, "5-fold OOF\nprobabilities\n$p_b^{\\mathrm{OOF}}$", color="#d5e8d4")

    mx, mw = 12.7, 2.2
    prune_c, per_c, thr_c = 7.05, 4.75, 2.45
    box(ax, (mx, prune_c - 0.75), mw, 1.5, "Skill-based\npruning ($\\delta$)", color="#d5e8d4")
    box(ax, (mx, per_c - 0.85), mw, 1.7, "Per-sample\nweighting\n($\\tau$, margins)", color="#d5e8d4")
    box(ax, (mx, thr_c - 0.75), mw, 1.5, "Threshold\ntuning ($\\Theta$, $J$)", color="#d5e8d4")

    rx, rw = 15.65, 2.15
    refit_c, pred_c = 6.25, 3.25
    box(ax, (rx, refit_c - 0.9), rw, 1.8, "Refit kept\nbases on full\ntraining fold", color="#fff2cc")
    box(ax, (rx, pred_c - 0.9), rw, 1.8, "Test-fold\nprediction\n$(\\hat{p}, \\hat{y}$ at $\\theta^*)$", color="#f8cecc")

    arrow(ax, (1.75, flow_y), (2.4, flow_y))
    bus1 = 4.75
    ax.plot([bus1, bus1], [base_c[-1], base_c[0]], color="#003366", linewidth=1.1)
    ax.plot([4.1, bus1], [flow_y, flow_y], color="#333", linewidth=1.3)
    for c in base_c:
        arrow(ax, (bus1, c), (bx, c))
    bus2 = 8.65
    ax.plot([bus2, bus2], [base_c[-1], base_c[0]], color="#003366", linewidth=1.1)
    for c in base_c:
        ax.plot([bx + bw, bus2], [c, c], color="#333", linewidth=1.3)
    arrow(ax, (bus2, flow_y), (ox, flow_y))

    src = (ox + ow, flow_y)
    arrow(ax, src, (mx, prune_c))
    arrow(ax, src, (mx, per_c))
    arrow(ax, src, (mx, thr_c))

    arrow(ax, (mx + mw, prune_c), (rx, refit_c + 0.4))
    arrow(ax, (mx + mw, per_c), (rx, refit_c - 0.4))
    arrow(ax, (mx + mw, thr_c), (rx, pred_c))
    arrow(ax, (rx + rw / 2, refit_c - 0.9), (rx + rw / 2, pred_c + 0.9))

    # No tight crop — keep exact FULL_WIDTH_IN for Word.
    fig.savefig(OUT / "hies_architecture.png", dpi=PNG_DPI, bbox_inches=None, pad_inches=0)
    fig.savefig(OUT / "hies_architecture.svg", bbox_inches=None, pad_inches=0)
    fig.savefig(OUT / "hies_architecture.pdf", bbox_inches=None, pad_inches=0)
    print(f"Wrote {OUT}/hies_architecture.[png|svg|pdf]")


if __name__ == "__main__":
    main()
