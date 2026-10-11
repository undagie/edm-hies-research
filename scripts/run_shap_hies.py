"""SHAP mean(|value|) for HIES via TreeExplainer on the kept XGBoost base.

Top-15 features per dataset and top-10 per sensitive group under
``artifacts/phase1_analysis/``.

Usage: python scripts/run_shap_hies.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold

from src.config import load_config
from src.datasets import load_dataset_bundle
from src.models.methods import build_method, make_preprocessor
from src.runner.experiment import _prepare_xy
from src.utils import silence_known_warnings

silence_known_warnings()
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "phase1_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def _extract_xgb_inner(hies_fitted) -> object:
    """Locate the kept XGBoost base inside a fitted HIES estimator."""
    for name, est in hies_fitted:
        if name != "xgb_spw":
            continue
        inner_cal = getattr(est, "calibrated_", None) or est
        cal_list = getattr(inner_cal, "calibrated_classifiers_", None)
        if cal_list:
            return cal_list[0].estimator
        return getattr(est, "estimator", est)
    return None


def _plot_top15(df: pd.DataFrame, title: str, save_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["feature"][::-1], df["mean_abs_shap"][::-1], color="tab:purple")
    ax.set_xlabel("Mean(|SHAP value|) — contribution to log-odds")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_stem.with_suffix(".png"), dpi=160)
    fig.savefig(save_stem.with_suffix(".pdf"))
    plt.close(fig)


def _per_group_top10(
    sv: np.ndarray, sensitive_test: np.ndarray, feat_names: np.ndarray, ds: str
) -> pd.DataFrame:
    rows: list[dict] = []
    for grp in np.unique(sensitive_test):
        mask = sensitive_test == grp
        if mask.sum() < 5:
            continue
        m = np.abs(sv[mask]).mean(axis=0)
        order = np.argsort(-m)[:10]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "dataset": ds,
                    "group": str(grp),
                    "rank": rank,
                    "feature": feat_names[idx] if idx < len(feat_names) else f"f{idx}",
                    "mean_abs_shap": float(m[idx]),
                }
            )
    return pd.DataFrame(rows)


def run_for_dataset(ds: str, cfg) -> None:
    print(f"\n=== {ds} ===")
    bundle = load_dataset_bundle(ds, cfg.data["raw_dir"], seed=cfg.seed, allow_synthetic=False)
    X, y, sens = _prepare_xy(
        bundle.frame,
        bundle.target_col,
        bundle.sensitive_col,
        leakage_columns=bundle.leakage_columns,
        enable_feature_interactions=cfg.data.get("enable_feature_interactions", False),
        max_interactions=cfg.data.get("max_feature_interactions", 0),
    )
    cwr = max((len(y) - y.sum()), 1) / max(y.sum(), 1)
    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=11)
    tr, te = next(cv.split(X, y))
    Xt, yt = X.iloc[tr], y.iloc[tr]
    Xv = X.iloc[te]
    sens_v = sens.iloc[te].astype(str).to_numpy()
    pre = make_preprocessor(Xt)
    feat_names = pre.fit(Xt).get_feature_names_out()
    Xt_t = pre.transform(Xt)
    Xv_t = pre.transform(Xv)
    if hasattr(Xt_t, "toarray"):
        Xt_t = Xt_t.toarray()
    if hasattr(Xv_t, "toarray"):
        Xv_t = Xv_t.toarray()

    spec = build_method(
        "hies",
        seed=11,
        class_weight_ratio=cwr,
        method_params=cfg.optimization["method_params"],
    )
    spec.estimator.fit(Xt_t, yt.values)
    xgb_inner = _extract_xgb_inner(spec.estimator.fitted_estimators_)
    if xgb_inner is None:
        print(f"  WARN: could not locate xgb_spw for {ds}; skipping.")
        return

    # Cap eval size for TreeExplainer runtime on larger folds.
    n_eval = min(300, Xv_t.shape[0])
    rng = np.random.default_rng(0)
    pick = rng.choice(Xv_t.shape[0], size=n_eval, replace=False)
    Xv_sub = Xv_t[pick]
    sens_sub = sens_v[pick]
    explainer = shap.TreeExplainer(xgb_inner)
    sv = explainer.shap_values(Xv_sub)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    # Some XGB versions emit (n, k, n_classes); keep positive class.
    if sv.ndim == 3 and sv.shape[-1] >= 2:
        sv = sv[..., 1]

    mean_abs = np.abs(sv).mean(axis=0)
    df = (
        pd.DataFrame(
            {
                "feature": feat_names[: len(mean_abs)],
                "mean_abs_shap": mean_abs,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
    )
    df.to_csv(OUT / f"shap_top15_{ds}.csv", index=False)
    _plot_top15(
        df,
        title=f"HIES (XGBoost base) — Top-15 features ({ds})",
        save_stem=OUT / f"shap_top15_{ds}",
    )
    print(df.head(15).to_string(index=False))

    pg = _per_group_top10(sv, sens_sub, feat_names, ds)
    if not pg.empty:
        pg.to_csv(OUT / f"shap_per_group_top10_{ds}.csv", index=False)
        print(f"  per-group SHAP saved ({pg['group'].nunique()} groups)")


def main() -> None:
    cfg = load_config("configs/default.yaml")
    for ds in cfg.data["dataset_order"]:
        run_for_dataset(ds, cfg)


if __name__ == "__main__":
    main()
