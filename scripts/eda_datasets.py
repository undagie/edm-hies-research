"""EDA for benchmark datasets; feature matrix matches experiment ``_prepare_xy``.

Writes tables/figures under ``artifacts/eda/``.

Usage: python scripts/eda_datasets.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.feature_selection import mutual_info_classif

from src.config import load_config
from src.datasets.loader import load_dataset_bundle
from src.runner.experiment import _prepare_xy
from src.utils import silence_known_warnings

silence_known_warnings()

sns.set_theme(style="whitegrid", context="notebook")


def _cramers_v(confusion: np.ndarray) -> float:
    """Cramér's V from contingency table."""
    confusion = np.asarray(confusion, dtype=float)
    n = confusion.sum()
    if n <= 0:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(confusion)
    if chi2 is None or np.isnan(chi2):
        return float("nan")
    r, k = confusion.shape
    denom = n * min(k - 1, r - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / denom))


def _feature_modes(cfg) -> list[tuple[str, bool, int]]:
    """Return (label, enable_interactions, max_interactions) variants."""
    en = bool(cfg.data.get("enable_feature_interactions", False))
    mx = int(cfg.data.get("max_feature_interactions", 0))
    modes: list[tuple[str, bool, int]] = [
        ("base_features", False, 0),
    ]
    if en and mx > 0:
        modes.append(("with_interactions", True, mx))
    return modes


def _numeric_categorical_split(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = [c for c in X.columns if X[c].dtype.kind in "if"]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _dataset_summary_row(
    dataset_name: str,
    mode_label: str,
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    bundle,
    n_splits: int,
) -> dict:
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    minority = 0 if n0 <= n1 else 1
    n_minor = min(n0, n1)
    n_major = max(n0, n1)
    imbalance_ratio = float(n_major / max(n_minor, 1))
    counts = y.value_counts()
    feasible = bool(counts.min() >= n_splits)
    k = n_splits
    # Stratified train fold ≈ (k-1)/k of full class counts.
    train_frac = (k - 1) / k
    approx_n0_train = int(np.floor(n0 * train_frac))
    approx_n1_train = int(np.floor(n1 * train_frac))
    numeric_cols, _cat = _numeric_categorical_split(X)
    sens_groups = sensitive.astype(str).nunique()
    sec_name = bundle.secondary_sensitive_col
    return {
        "dataset": dataset_name,
        "feature_mode": mode_label,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "n_numeric": len(numeric_cols),
        "n_categorical": X.shape[1] - len(numeric_cols),
        "class_0_count": n0,
        "class_1_count": n1,
        "minority_class": minority,
        "imbalance_ratio": imbalance_ratio,
        "approx_class0_per_train_fold": approx_n0_train,
        "approx_class1_per_train_fold": approx_n1_train,
        "approx_minority_per_train_fold": min(approx_n0_train, approx_n1_train),
        "stratified_kfold_feasible": feasible,
        "n_splits_configured": n_splits,
        "sensitive_cardinality": sens_groups,
        "secondary_sensitive": sec_name if sec_name and sec_name in bundle.frame.columns else "",
    }


def _missingness_long(dataset_name: str, mode_label: str, X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        s = X[col]
        if hasattr(s, "isna"):
            na = int(s.isna().sum())
        else:
            na = int(pd.isna(s).sum())
        pct = 100.0 * na / max(len(X), 1)
        rows.append(
            {
                "dataset": dataset_name,
                "feature_mode": mode_label,
                "column": col,
                "n_missing": na,
                "pct_missing": round(pct, 4),
            }
        )
    return pd.DataFrame(rows)


def _cardinality_long(dataset_name: str, mode_label: str, X: pd.DataFrame) -> pd.DataFrame:
    _, cat_cols = _numeric_categorical_split(X)
    rows = []
    for col in cat_cols:
        vc = X[col].astype(str).value_counts()
        top = vc.head(5).to_dict()
        top_str = "; ".join(f"{k}:{v}" for k, v in list(top.items())[:5])
        rows.append(
            {
                "dataset": dataset_name,
                "feature_mode": mode_label,
                "column": col,
                "n_unique": int(X[col].astype(str).nunique()),
                "top_5_values": top_str[:500],
            }
        )
    return pd.DataFrame(rows)


def _associations(dataset_name: str, mode_label: str, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    y_arr = np.asarray(y).astype(int)
    numeric_cols, cat_cols = _numeric_categorical_split(X)
    rows: list[dict] = []

    for col in numeric_cols:
        x = pd.to_numeric(X[col], errors="coerce")
        mask = np.isfinite(x.to_numpy(dtype=float))
        if mask.sum() < 10:
            rows.append(
                {
                    "dataset": dataset_name,
                    "feature_mode": mode_label,
                    "column": col,
                    "kind": "numeric",
                    "point_biserial_r": np.nan,
                    "point_biserial_p": np.nan,
                    "mutual_info": np.nan,
                    "cramers_v": np.nan,
                    "chi2_pvalue": np.nan,
                }
            )
            continue
        xv = x.to_numpy(dtype=float)[mask]
        yv = y_arr[mask]
        if np.nanstd(xv) < 1e-12:
            r, p = np.nan, np.nan
            mi = 0.0
        else:
            r, p = pointbiserialr(yv, xv)
            mi = float(
                mutual_info_classif(
                    xv.reshape(-1, 1),
                    yv,
                    discrete_features=False,
                    random_state=0,
                )[0]
            )
        rows.append(
            {
                "dataset": dataset_name,
                "feature_mode": mode_label,
                "column": col,
                "kind": "numeric",
                "point_biserial_r": float(r) if not np.isnan(r) else np.nan,
                "point_biserial_p": float(p) if not np.isnan(p) else np.nan,
                "mutual_info": mi,
                "cramers_v": np.nan,
                "chi2_pvalue": np.nan,
            }
        )

    for col in cat_cols:
        ct = pd.crosstab(X[col].astype(str), y_arr)
        if ct.size == 0 or ct.shape[0] < 2:
            cv = np.nan
            pval = np.nan
        else:
            cv = _cramers_v(ct.values)
            try:
                chi2, p, _, _ = chi2_contingency(ct.values)
                _ = chi2
                pval = float(p)
            except ValueError:
                pval = np.nan
        rows.append(
            {
                "dataset": dataset_name,
                "feature_mode": mode_label,
                "column": col,
                "kind": "categorical",
                "point_biserial_r": np.nan,
                "point_biserial_p": np.nan,
                "mutual_info": np.nan,
                "cramers_v": cv,
                "chi2_pvalue": pval,
            }
        )

    return pd.DataFrame(rows)


def _sensitive_label_rates(
    dataset_name: str,
    bundle,
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    rows = []
    for role, col in (
        ("primary", bundle.sensitive_col),
        ("secondary", bundle.secondary_sensitive_col),
    ):
        if not col or col not in df.columns:
            continue
        sub = df[[col, target_col]].dropna()
        sub[col] = sub[col].astype(str)
        for g, gdf in sub.groupby(col):
            n = len(gdf)
            pr = float(gdf[target_col].astype(int).mean())
            rows.append(
                {
                    "dataset": dataset_name,
                    "role": role,
                    "attribute": col,
                    "group": str(g),
                    "n": n,
                    "positive_rate": round(pr, 6),
                }
            )
    return pd.DataFrame(rows)


def _plot_missingness(df_miss: pd.DataFrame, dataset: str, mode_label: str, out_path: Path) -> None:
    sub = df_miss[(df_miss["dataset"] == dataset) & (df_miss["feature_mode"] == mode_label)].copy()
    sub = sub.sort_values("pct_missing", ascending=False).head(25)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sub["column"].astype(str)[::-1], sub["pct_missing"][::-1], color="steelblue")
    ax.set_xlabel("Percent missing (%)")
    ax.set_title(f"Top 25 missingness — {dataset} ({mode_label})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_label_rates(df_rates: pd.DataFrame, dataset: str, out_path: Path) -> None:
    sub = df_rates[df_rates["dataset"] == dataset]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    primary = sub[sub["role"] == "primary"]
    if not primary.empty:
        x = np.arange(len(primary))
        ax.bar(x, primary["positive_rate"], color="coral", label="positive rate")
        ax.set_xticks(x)
        ax.set_xticklabels(primary["group"], rotation=45, ha="right")
        ax.set_ylabel("Positive rate (target=1)")
        ax.set_title(f"Label rate by sensitive group — {dataset}")
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_corr_heatmap(X: pd.DataFrame, dataset: str, mode_label: str, y: np.ndarray, out_path: Path) -> None:
    numeric_cols, _ = _numeric_categorical_split(X)
    if len(numeric_cols) < 2:
        return
    y_arr = np.asarray(y).astype(int)
    scores = []
    for col in numeric_cols:
        x = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 10 or np.nanstd(x[mask]) < 1e-12:
            scores.append((col, 0.0))
            continue
        r, _ = pointbiserialr(y_arr[mask], x[mask])
        scores.append((col, abs(float(r)) if not np.isnan(r) else 0.0))
    scores.sort(key=lambda t: -t[1])
    top_k = min(15, len(scores))
    keep = [c for c, _ in scores[:top_k]]
    if len(keep) < 2:
        return
    sub = X[keep].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="pearson", min_periods=10)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, cmap="vlag", center=0.0, vmin=-1, vmax=1, square=True)
    ax.set_title(f"Numeric Pearson correlation (top {top_k} by |r(y,x)|) — {dataset} ({mode_label})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_top_numeric_violin(X: pd.DataFrame, y: np.ndarray, dataset: str, mode_label: str, out_path: Path) -> None:
    numeric_cols, _ = _numeric_categorical_split(X)
    y_arr = np.asarray(y).astype(int)
    ranked: list[tuple[str, float]] = []
    for col in numeric_cols:
        x = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 10 or np.nanstd(x[mask]) < 1e-12:
            continue
        r, _ = pointbiserialr(y_arr[mask], x[mask])
        if not np.isnan(r):
            ranked.append((col, abs(float(r))))
    ranked.sort(key=lambda t: -t[1])
    top = [c for c, _ in ranked[:3]]
    if not top:
        return
    plot_df = X[top].copy()
    plot_df["_y"] = y_arr
    fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 4), squeeze=False)
    for ax, col in zip(axes[0], top):
        d0 = pd.to_numeric(plot_df.loc[plot_df["_y"] == 0, col], errors="coerce").dropna()
        d1 = pd.to_numeric(plot_df.loc[plot_df["_y"] == 1, col], errors="coerce").dropna()
        parts = ax.violinplot([d0, d1], positions=[0, 1], showmeans=True, showmedians=True)
        for b in parts["bodies"]:
            b.set_alpha(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["y=0", "y=1"])
        ax.set_title(col[:40])
    fig.suptitle(f"Top numeric features by |point-biserial| — {dataset} ({mode_label})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _df_to_simple_markdown(df: pd.DataFrame) -> str:
    """Markdown table without optional ``tabulate`` dependency."""
    cols = list(df.columns)
    esc = lambda x: str(x).replace("|", "\\|")
    header = "| " + " | ".join(esc(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(esc(row[c]) for c in cols) + " |")
    return "\n".join([header, sep, *rows])


def _build_report(
    summary: pd.DataFrame,
    out_md: Path,
    figure_rel_paths: list[str],
) -> None:
    lines = [
        "# EDA report (benchmark feature matrix)",
        "",
        "Feature definitions match `src.runner.experiment._prepare_xy` (no leakage columns).",
        "",
        "## Dataset summary",
        "",
        _df_to_simple_markdown(summary),
        "",
        "## Figures",
        "",
    ]
    for rel in figure_rel_paths:
        lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## Implications for HIES (`hies`)",
            "",
            "These are **hypothesis-oriented** next steps tied to `_HIESAdaptive` knobs ",
            "(`auto_config`, `margin_floor`, `min_keep`, `n_train_threshold`, calibration, threshold grids). ",
            "They are not guarantees of leaderboard dominance.",
            "",
            "### `uci_student` (typically small N, high variance across seeds)",
            "",
            "- **Hypothesis 1:** Low effective sample per fold increases OOF noise; prefer **lower `cv_splits`** in sweep ",
            "  or stronger regularization in base learners before ensemble fusion.",
            "- **Hypothesis 2:** If univariate signal is spread across many weak numeric features, **soft voting / margin weighting** ",
            "  may outperform stacking-like fusion — align `temperature` and `margin_floor` with observed class separation.",
            "- **Hypothesis 3:** Tighten **threshold grid** around the empirical optimal region from fold-wise validation ",
            "  if positive rates differ sharply by sensitive group (fairness–utility tradeoff).",
            "- **Hypothesis 4:** If categorical cardinality is moderate, ensure **sigmoid vs isotonic** calibration choice ",
            "  matches dataset size bucket (`auto_config` / `n_train_threshold`) to avoid brittle probabilities.",
            "",
            "### `uci_dropout_success` (large N, heavy imbalance context)",
            "",
            "- **Hypothesis 1:** Large train sets favour stable calibration; revisit **`n_train_threshold`** and ",
            "  **`calibration_method`** buckets so large-N regimes do not default to unstable high-variance calibrators.",
            "- **Hypothesis 2:** High cardinality categoricals inflate one-hot dimensionality — tree ensembles dominate; ",
            "  consider **prune_delta / min_keep** so consistently weak bases do not dilute fused probabilities.",
            "- **Hypothesis 3:** If missingness concentrates in specific columns, imbalance-adaptive bases differ in ",
            "  imputation sensitivity — compare RF vs boosted bases under the same preprocessor (already shared).",
            "- **Hypothesis 4:** Use **`imbalanced_mcc_boost`** in `auto_config` when imbalance ratio exceeds `imbalance_split` ",
            "  to align internal threshold tuning with MCC-sensitive regions of the ROC.",
            "",
            "### `xapi` (heterogeneous tabular / behaviour proxies)",
            "",
            "- **Hypothesis 1:** When complementary bases disagree (high margin variance), **per-sample weighting** helps — ",
            "  tune **`margin_floor`** upward if margins collapse for the minority class.",
            "- **Hypothesis 2:** If correlation among top numerics is low, heterogeneous ensembles capture complementary cuts; ",
            "  keep **`min_keep` ≥ 2** but prune bases with consistently low OOF macro-F1 (`prune_delta`).",
            "",
            "### Cross-dataset",
            "",
            "- Prefer **dataset-conditioned `auto_config`** (small vs large split) rather than one global YAML when ",
            "  summaries diverge materially in `n_rows`, `imbalance_ratio`, and categorical footprint.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for benchmark datasets.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic CSV fallback (default: false; fail if raw missing).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg.data["raw_dir"]
    order = list(cfg.data.get("dataset_order", []))
    n_splits = int(cfg.n_splits)
    allow_syn = bool(args.allow_synthetic or cfg.data.get("use_synthetic_if_missing", False))

    out_root = Path("artifacts/eda")
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    missings = []
    cards = []
    assocs = []
    sens_rates = []

    figure_refs: list[str] = []

    for dataset_name in order:
        bundle = load_dataset_bundle(
            dataset_name,
            raw_dir,
            seed=int(cfg.seed),
            allow_synthetic=allow_syn,
        )
        df = bundle.frame
        target_col = bundle.target_col
        sens_col = bundle.sensitive_col

        for mode_label, en_int, max_int in _feature_modes(cfg):
            X, y, sensitive = _prepare_xy(
                df,
                target_col,
                sens_col,
                leakage_columns=bundle.leakage_columns,
                enable_feature_interactions=en_int,
                max_interactions=max_int,
            )
            summaries.append(
                _dataset_summary_row(
                    dataset_name,
                    mode_label,
                    X,
                    y,
                    sensitive,
                    bundle,
                    n_splits,
                )
            )
            mlong = _missingness_long(dataset_name, mode_label, X)
            missings.append(mlong)
            cards.append(_cardinality_long(dataset_name, mode_label, X))
            assocs.append(_associations(dataset_name, mode_label, X, y.values))

            rel_miss = f"artifacts/eda/figures/missingness_{dataset_name}_{mode_label}.png"
            _plot_missingness(mlong, dataset_name, mode_label, Path(rel_miss))
            figure_refs.append(rel_miss)

            rel_corr = f"artifacts/eda/figures/corr_{dataset_name}_{mode_label}.png"
            _plot_corr_heatmap(X, dataset_name, mode_label, y.values, Path(rel_corr))
            figure_refs.append(rel_corr)

            rel_viol = f"artifacts/eda/figures/top_numeric_{dataset_name}_{mode_label}.png"
            _plot_top_numeric_violin(X, y.values, dataset_name, mode_label, Path(rel_viol))
            figure_refs.append(rel_viol)

        sr = _sensitive_label_rates(dataset_name, bundle, df, target_col)
        sens_rates.append(sr)
        rel_lr = f"artifacts/eda/figures/label_rate_{dataset_name}.png"
        _plot_label_rates(sr, dataset_name, Path(rel_lr))
        figure_refs.append(rel_lr)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_root / "dataset_summary.csv", index=False)
    pd.concat(missings, ignore_index=True).to_csv(out_root / "feature_missingness.csv", index=False)
    pd.concat(cards, ignore_index=True).to_csv(out_root / "feature_cardinality.csv", index=False)
    pd.concat(assocs, ignore_index=True).to_csv(out_root / "univariate_association.csv", index=False)
    pd.concat(sens_rates, ignore_index=True).to_csv(out_root / "sensitive_label_rates.csv", index=False)

    report_path = out_root / "eda_report.md"
    _build_report(summary_df, report_path, sorted(set(figure_refs)))

    print(f"Wrote {out_root / 'dataset_summary.csv'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
