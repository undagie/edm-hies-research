"""Friedman + Nemenyi post-hoc + Critical Difference diagrams (Demsar 2006).

Subjects are (dataset, seed, fold) triples; methods are treatments.
Writes tables and CD figures under ``artifacts/phase1_analysis/``.

Usage: python scripts/analyze_friedman_nemenyi.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata
import scikit_posthocs as sp

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fig_style import DATASET_LABELS, FULL_WIDTH_IN, PNG_DPI, apply_ijies_style
from src.reporting.method_names import manuscript_label

RAW = ROOT / "artifacts" / "results" / "results_raw.csv"
OUT = ROOT / "artifacts" / "phase1_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def _pivot_metric(raw: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    """Long-form (dataset, seed, fold, method, metric) -> wide (rows x methods)."""
    raw = raw.copy()
    raw["block"] = (
        raw["dataset"].astype(str)
        + "::seed=" + raw["seed"].astype(str)
        + "::fold=" + raw["fold"].astype(str)
    )
    wide = raw.pivot_table(
        index="block", columns="method", values=metric, aggfunc="mean"
    )
    return wide.dropna(axis=0, how="any")


def friedman_test(wide: pd.DataFrame) -> tuple[float, float]:
    arrays = [wide[m].to_numpy() for m in wide.columns]
    stat, p = friedmanchisquare(*arrays)
    return float(stat), float(p)


def _critical_difference(num_methods: int, num_subjects: int, alpha: float = 0.05) -> float:
    """Nemenyi CD = q_alpha * sqrt(k*(k+1)/(6N)); q from Demsar 2006 Table 5."""
    q_alpha_05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
        14: 3.354, 15: 3.391, 20: 3.532, 25: 3.628, 30: 3.701,
    }
    q = q_alpha_05.get(num_methods)
    if q is None:
        keys = sorted(q_alpha_05)
        lo = max(k for k in keys if k <= num_methods) if any(k <= num_methods for k in keys) else min(keys)
        hi = min(k for k in keys if k >= num_methods) if any(k >= num_methods for k in keys) else max(keys)
        if lo == hi:
            q = q_alpha_05[lo]
        else:
            f = (num_methods - lo) / (hi - lo)
            q = q_alpha_05[lo] + f * (q_alpha_05[hi] - q_alpha_05[lo])
    cd = q * np.sqrt(num_methods * (num_methods + 1) / (6.0 * num_subjects))
    return float(cd)


def cd_diagram(
    wide: pd.DataFrame,
    title: str,
    save_pdf: Path,
    save_png: Path | None = None,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> dict:
    """Demsar-style CD diagram: avg ranks on axis; bars join non-significant groups."""
    ranks = wide.apply(
        lambda row: rankdata(-row.to_numpy() if higher_is_better else row.to_numpy()),
        axis=1,
        result_type="broadcast",
    )
    avg_rank = ranks.mean(axis=0).sort_values()
    methods = avg_rank.index.tolist()
    ranks_sorted = avg_rank.values
    k = len(methods)
    n = len(wide)
    cd = _critical_difference(k, n, alpha)

    fig_height = 1.6 + 0.20 * k
    fig, ax = plt.subplots(figsize=(8.8, fig_height))
    ax.set_xlim(k + 0.5, 0.5)
    ax.set_ylim(-k - 1, 2)
    ax.invert_xaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticks(list(range(1, k + 1)))
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title(title, pad=20)

    cd_y = 1.4
    ax.plot([1, 1 + cd], [cd_y, cd_y], color="black", linewidth=2)
    ax.plot([1, 1], [cd_y - 0.15, cd_y + 0.15], color="black", linewidth=2)
    ax.plot([1 + cd, 1 + cd], [cd_y - 0.15, cd_y + 0.15], color="black", linewidth=2)
    ax.text(1 + cd / 2.0, cd_y + 0.3, f"CD = {cd:.3f}  (alpha={alpha})", ha="center", fontsize=9)

    label_y_left = -1
    label_y_right = -1
    left_methods = []
    right_methods = []
    midpoint = (k + 1) / 2.0
    for i, m in enumerate(methods):
        r = ranks_sorted[i]
        if r <= midpoint:
            right_methods.append((r, m))
        else:
            left_methods.append((r, m))

    for j, (r, m) in enumerate(right_methods):
        y_label = label_y_right - 0.6 * j
        ax.plot([r, r], [0, y_label + 0.1], color="black", linewidth=0.8)
        ax.plot([r, 0.7], [y_label + 0.1, y_label + 0.1], color="black", linewidth=0.8)
        ax.text(0.6, y_label + 0.1, f"{manuscript_label(m)} ({r:.2f})", ha="left", va="center", fontsize=9)

    for j, (r, m) in enumerate(left_methods):
        y_label = label_y_left - 0.6 * j
        ax.plot([r, r], [0, y_label + 0.1], color="black", linewidth=0.8)
        ax.plot([r, k + 0.3], [y_label + 0.1, y_label + 0.1], color="black", linewidth=0.8)
        ax.text(k + 0.4, y_label + 0.1, f"{manuscript_label(m)} ({r:.2f})", ha="right", va="center", fontsize=9)

    bar_y = -0.4
    drawn_bars: list[tuple[float, float]] = []
    for i in range(k):
        right = i
        for j in range(i + 1, k):
            if ranks_sorted[j] - ranks_sorted[i] < cd:
                right = j
            else:
                break
        if right > i:
            seg = (ranks_sorted[i], ranks_sorted[right])
            if any(a <= seg[0] and seg[1] <= b for a, b in drawn_bars):
                continue
            drawn_bars.append(seg)
            ax.plot(
                [seg[0] - 0.05, seg[1] + 0.05],
                [bar_y - 0.18 * len(drawn_bars), bar_y - 0.18 * len(drawn_bars)],
                color="black",
                linewidth=3,
                solid_capstyle="butt",
            )

    fig.tight_layout()
    fig.savefig(save_pdf)
    if save_png is not None:
        fig.savefig(save_png, dpi=PNG_DPI)
    plt.close(fig)
    return {
        "cd": cd,
        "n_subjects": int(n),
        "n_methods": int(k),
        "avg_ranks": avg_rank.to_dict(),
    }


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"Missing {RAW}; run main benchmark first.")
    raw = pd.read_csv(RAW)
    methods = sorted(raw["method"].unique())
    print(f"Loaded {len(raw)} rows, {len(methods)} methods")

    friedman_rows: list[dict] = []
    nemenyi_long: list[dict] = []
    panels: list[tuple[str, pd.DataFrame, dict]] = []

    for ds in sorted(raw["dataset"].unique()):
        sub = raw[raw["dataset"] == ds]
        wide = _pivot_metric(sub, "f1_macro")
        stat, p = friedman_test(wide)
        friedman_rows.append(
            {
                "dataset": ds,
                "n_subjects": int(len(wide)),
                "n_methods": int(len(methods)),
                "chi2": stat,
                "p_value": p,
                "significant_alpha_05": bool(p < 0.05),
            }
        )
        nemenyi = sp.posthoc_nemenyi_friedman(wide.values)
        nemenyi.index = wide.columns
        nemenyi.columns = wide.columns
        for m1 in nemenyi.index:
            for m2 in nemenyi.columns:
                if m1 < m2:
                    nemenyi_long.append(
                        {
                            "dataset": ds,
                            "method_a": m1,
                            "method_b": m2,
                            "nemenyi_p": float(nemenyi.loc[m1, m2]),
                            "significant_alpha_05": bool(nemenyi.loc[m1, m2] < 0.05),
                        }
                    )
        meta = cd_diagram(
            wide,
            title=f"Critical Difference — {ds}  (Friedman p={p:.3g})",
            save_pdf=OUT / f"cd_diagram_{ds}.pdf",
            save_png=OUT / f"cd_diagram_{ds}.png",
        )
        panels.append((ds, wide, meta))
        print(f"{ds}: chi2={stat:.2f} p={p:.3g} CD={meta['cd']:.3f} n={meta['n_subjects']}")

    pd.DataFrame(friedman_rows).to_csv(OUT / "table_friedman.csv", index=False)
    pd.DataFrame(nemenyi_long).to_csv(OUT / "table_nemenyi_pvalues.csv", index=False)

    pooled = _pivot_metric(raw, "f1_macro")
    stat_g, p_g = friedman_test(pooled)
    meta_global = cd_diagram(
        pooled,
        title=f"Critical Difference — pooled across datasets  (Friedman p={p_g:.3g})",
        save_pdf=OUT / "cd_diagram_all_folds.pdf",
        save_png=OUT / "cd_diagram_all_folds.png",
    )
    print(
        f"pooled: chi2={stat_g:.2f} p={p_g:.3g} CD={meta_global['cd']:.3f} "
        f"n={meta_global['n_subjects']}"
    )
    pd.DataFrame(
        friedman_rows + [{
            "dataset": "__pooled__",
            "n_subjects": int(meta_global["n_subjects"]),
            "n_methods": int(meta_global["n_methods"]),
            "chi2": stat_g,
            "p_value": p_g,
            "significant_alpha_05": bool(p_g < 0.05),
        }]
    ).to_csv(OUT / "table_friedman.csv", index=False)

    # Per-dataset avg ranks as one print-width dot plot; CD identical across ds.
    apply_ijies_style()
    ds_ranks: dict[str, pd.Series] = {}
    for ds, wide, meta in panels:
        ranks = wide.apply(lambda r: rankdata(-r.to_numpy()), axis=1, result_type="broadcast")
        ds_ranks[ds] = ranks.mean(axis=0)
    rank_frame = pd.DataFrame(ds_ranks)
    order = rank_frame.mean(axis=1).sort_values().index.tolist()
    rank_frame = rank_frame.loc[order]
    k = len(order)
    cd_val = panels[0][2]["cd"]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 0.205 * k + 0.6))
    y_pos = np.arange(k)[::-1]
    ds_markers = {"uci_student": "o", "uci_dropout_success": "s", "xapi": "^"}
    ds_colors = {"uci_student": "tab:blue", "uci_dropout_success": "tab:green", "xapi": "tab:orange"}
    for ds in rank_frame.columns:
        ax.scatter(
            rank_frame[ds].to_numpy(),
            y_pos,
            marker=ds_markers.get(ds, "o"),
            s=30,
            facecolors="none",
            edgecolors=ds_colors.get(ds, "gray"),
            linewidths=1.3,
            label=DATASET_LABELS.get(ds, ds),
            zorder=3,
        )
    for y in y_pos:
        ax.axhline(y, color="0.88", linewidth=0.6, zorder=1)
    hies_idx = order.index("hies") if "hies" in order else None
    labels = [manuscript_label(m) for m in order]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    if hies_idx is not None:
        ax.get_yticklabels()[hies_idx].set_fontweight("bold")
    ax.set_xlabel("Average rank over 60 seed-fold blocks (lower is better)")
    ax.set_xlim(0.5, k + 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cd_y = -0.4
    ax.plot([k - cd_val, k], [cd_y, cd_y], color="black", linewidth=1.6, clip_on=False)
    for x in (k - cd_val, k):
        ax.plot([x, x], [cd_y - 0.22, cd_y + 0.22], color="black", linewidth=1.6, clip_on=False)
    ax.text(k - cd_val / 2, cd_y + 0.35, f"CD = {cd_val:.2f}", ha="center", va="bottom")
    ax.set_ylim(cd_y - 0.5, k - 0.4)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "cd_diagram_per_dataset.pdf", bbox_inches="tight")
    fig.savefig(OUT / "cd_diagram_per_dataset.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Outputs in {OUT}")


if __name__ == "__main__":
    main()
