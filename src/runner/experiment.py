"""Paired seed×fold benchmark loop: fit, score, aggregate, write tables."""

from __future__ import annotations

import logging
import os
import time
from copy import deepcopy
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.datasets import load_dataset_bundle
from src.evaluation import (
    compute_calibration,
    compute_calibration_per_group,
    compute_fairness,
    compute_performance,
)
from src.reporting.best_per_dataset import select_best_per_dataset
from src.models.methods import (
    build_method,
    compute_fair_reweighting,
    find_group_thresholds,
    make_preprocessor,
)


def _prepare_xy(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    leakage_columns: list[str] | None = None,
    enable_feature_interactions: bool = False,
    max_interactions: int = 0,
):
    y = df[target_col].astype(int)
    sensitive = df[sensitive_col].astype(str)
    drop_cols = {target_col, sensitive_col}
    if leakage_columns:
        drop_cols.update(leakage_columns)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if enable_feature_interactions and max_interactions > 0:
        numeric_cols = [c for c in X.columns if X[c].dtype.kind in "if"]
        pair_count = 0
        for i, left in enumerate(numeric_cols):
            for right in numeric_cols[i + 1 :]:
                X[f"{left}_x_{right}"] = X[left] * X[right]
                pair_count += 1
                if pair_count >= max_interactions:
                    break
            if pair_count >= max_interactions:
                break
    return X, y, sensitive


def _mlflow_enabled(config) -> bool:
    return hasattr(config, "outputs") and isinstance(config.outputs, dict)


def _resolve_optimization_config(config) -> dict:
    defaults = {
        "objective_weights": {
            "f1_macro": 0.65,
            "mcc": 0.35,
            "dp_diff": 0.2,
            "eod_diff": 0.1,
        },
        "threshold_grid": [0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
        "enable_threshold_optimization": True,
        "threshold_validation_fraction": 0.2,
        "max_dp_increase": 0.03,
        "max_eod_increase": 0.03,
        "threshold_methods": ["xgboost", "calibrated_xgboost"],
        "method_params": {},
        "tradeoff_baseline_method": "xgboost",
        "tradeoff_target_method": "calibrated_xgboost",
    }
    raw = getattr(config, "optimization", None)
    if not isinstance(raw, dict):
        return defaults
    merged = defaults.copy()
    merged.update(raw)
    if "objective_weights" in raw and isinstance(raw["objective_weights"], dict):
        weights = defaults["objective_weights"].copy()
        weights.update(raw["objective_weights"])
        merged["objective_weights"] = weights
    return merged


def _compute_joint_score(perf: dict[str, float], fair: dict[str, float], weights: dict[str, float]) -> float:
    return float(
        weights.get("f1_macro", 0.0) * perf["f1_macro"]
        + weights.get("mcc", 0.0) * perf["mcc"]
        - weights.get("dp_diff", 0.0) * fair["demographic_parity_diff"]
        - weights.get("eod_diff", 0.0) * fair["equal_opportunity_diff"]
    )


def _guardrail_pass(
    fair: dict[str, float],
    baseline_dp: float | None,
    baseline_eod: float | None,
    max_dp_increase: float,
    max_eod_increase: float,
) -> bool:
    if baseline_dp is None or baseline_eod is None:
        return True
    return bool(
        fair["demographic_parity_diff"] <= baseline_dp + max_dp_increase
        and fair["equal_opportunity_diff"] <= baseline_eod + max_eod_increase
    )


def _find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    threshold_grid: list[float],
    weights: dict[str, float],
    baseline_dp: float | None = None,
    baseline_eod: float | None = None,
    max_dp_increase: float = 0.03,
    max_eod_increase: float = 0.03,
) -> float:
    best_threshold = 0.5
    best_score = -float("inf")
    for threshold in threshold_grid:
        y_pred = (y_prob >= threshold).astype(int)
        perf = compute_performance(y_true, y_pred, y_prob)
        fair = compute_fairness(y_true, y_pred, sensitive)
        score = _compute_joint_score(perf=perf, fair=fair, weights=weights)
        if not _guardrail_pass(
            fair=fair,
            baseline_dp=baseline_dp,
            baseline_eod=baseline_eod,
            max_dp_increase=max_dp_increase,
            max_eod_increase=max_eod_increase,
        ):
            score -= 1.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _apply_fairness_guardrails(raw: pd.DataFrame, max_dp_increase: float, max_eod_increase: float) -> pd.DataFrame:
    baseline = raw[raw["method"] == "xgboost"][
        ["dataset", "seed", "fold", "demographic_parity_diff", "equal_opportunity_diff"]
    ].rename(
        columns={
            "demographic_parity_diff": "baseline_dp_diff",
            "equal_opportunity_diff": "baseline_eod_diff",
        }
    )
    if baseline.empty:
        out = raw.copy()
        out["fairness_guardrail_pass"] = True
        return out
    out = raw.merge(baseline, on=["dataset", "seed", "fold"], how="left")
    out["fairness_guardrail_pass"] = (
        (out["demographic_parity_diff"] <= out["baseline_dp_diff"] + float(max_dp_increase))
        & (out["equal_opportunity_diff"] <= out["baseline_eod_diff"] + float(max_eod_increase))
    )
    out.loc[out["method"] == "xgboost", "fairness_guardrail_pass"] = True
    out["fairness_guardrail_pass"] = out["fairness_guardrail_pass"].fillna(True)
    return out


def _apply_method_resampling(spec, X_train_t, y_train: pd.Series):
    X_fit, y_fit = X_train_t, y_train.values
    if spec.resampler is None:
        return X_fit, y_fit
    return spec.resampler.fit_resample(X_train_t, y_train)


def _setup_mlflow(config, run_id: str) -> None:
    if not _mlflow_enabled(config):
        return
    mlflow_cfg = config.outputs.get("mlflow", {})
    configured_uri = mlflow_cfg.get("tracking_uri")
    if configured_uri:
        if configured_uri.startswith("file:"):
            raw_path = configured_uri.replace("file:", "", 1)
            abs_path = Path(raw_path).resolve()
            tracking_uri = abs_path.as_uri()
        elif configured_uri.startswith("sqlite:///") and not configured_uri.startswith("sqlite:////"):
            db_path = configured_uri.replace("sqlite:///", "", 1)
            tracking_uri = f"sqlite:///{Path(db_path).resolve().as_posix()}"
        else:
            tracking_uri = configured_uri
    else:
        default_db = (Path(config.outputs["root"]) / "mlflow.db").resolve().as_posix()
        tracking_uri = f"sqlite:///{default_db}"
    experiment_name = mlflow_cfg.get("experiment_name", "edm-tabular-benchmark")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        os.environ.pop("MLFLOW_RUN_ID", None)
        mlflow.start_run(run_name=f"benchmark_{run_id}")
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("n_splits", str(config.n_splits))
        mlflow.log_param("seed", config.seed)
        mlflow.log_param("seeds", ",".join(str(s) for s in config.seeds))
        mlflow.log_param("methods", ",".join(config.methods))
        mlflow.log_param("dataset_order", ",".join(config.data["dataset_order"]))
    except Exception:
        if mlflow.active_run() is not None:
            mlflow.end_run()


def _log_mlflow_fold_metrics(row: dict) -> None:
    if mlflow.active_run() is None:
        return
    step = int(row["seed"]) * 100 + int(row["fold"])
    base = f"{row['dataset']}/{row['method']}"
    mlflow.log_metric(f"{base}/f1_macro", float(row["f1_macro"]), step=step)
    mlflow.log_metric(f"{base}/mcc", float(row["mcc"]), step=step)
    mlflow.log_metric(f"{base}/pr_auc", float(row["pr_auc"]), step=step)
    mlflow.log_metric(f"{base}/balanced_accuracy", float(row["balanced_accuracy"]), step=step)
    mlflow.log_metric(
        f"{base}/demographic_parity_diff",
        float(row["demographic_parity_diff"]),
        step=step,
    )
    mlflow.log_metric(
        f"{base}/equal_opportunity_diff",
        float(row["equal_opportunity_diff"]),
        step=step,
    )


def _build_fairness_table(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["dataset", "method"], as_index=False)
        .agg(
            dp_diff_mean=("demographic_parity_diff", "mean"),
            dp_diff_std=("demographic_parity_diff", "std"),
            eod_diff_mean=("equal_opportunity_diff", "mean"),
            eod_diff_std=("equal_opportunity_diff", "std"),
        )
        .sort_values(["dataset", "dp_diff_mean"], ascending=[True, True])
    )


def _build_significance_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, sub in raw.groupby("dataset"):
        baseline = sub[sub["method"] == "xgboost"]["f1_macro"]
        if baseline.empty:
            continue
        base_vals = baseline.to_numpy()
        for method_name, method_sub in sub.groupby("method"):
            vals = method_sub["f1_macro"].to_numpy()
            n = min(len(base_vals), len(vals))
            if n == 0:
                continue
            diff = vals[:n] - base_vals[:n]
            effect = float(np.mean(diff))
            std = float(np.std(diff, ddof=1)) if n > 1 else 0.0
            if n > 1 and not np.allclose(diff, 0.0):
                p_value = float(wilcoxon(vals[:n], base_vals[:n], zero_method="wilcox", alternative="two-sided").pvalue)
            else:
                p_value = 1.0
            rows.append(
                {
                    "dataset": dataset,
                    "method": method_name,
                    "baseline_method": "xgboost",
                    "mean_delta_f1_macro": effect,
                    "effect_size_d": effect / std if std > 0 else 0.0,
                    "p_value": p_value,
                }
            )
    columns = [
        "dataset",
        "method",
        "baseline_method",
        "mean_delta_f1_macro",
        "effect_size_d",
        "p_value",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    mask_non_baseline = frame["method"] != "xgboost"
    if mask_non_baseline.any():
        pvals = frame.loc[mask_non_baseline, "p_value"].to_numpy()
        _, holm_p, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        _, bh_p, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        frame.loc[mask_non_baseline, "p_value_holm"] = holm_p
        frame.loc[mask_non_baseline, "p_value_bh"] = bh_p
        frame.loc[mask_non_baseline, "significant_holm"] = holm_p < 0.05
        frame.loc[mask_non_baseline, "significant_bh"] = bh_p < 0.05
    frame.loc[~mask_non_baseline, "p_value_holm"] = frame.loc[~mask_non_baseline, "p_value"]
    frame.loc[~mask_non_baseline, "p_value_bh"] = frame.loc[~mask_non_baseline, "p_value"]
    frame.loc[~mask_non_baseline, "significant_holm"] = False
    frame.loc[~mask_non_baseline, "significant_bh"] = False
    return frame.sort_values(["dataset", "p_value"], ascending=[True, True])


def _build_ablation_table(_summary: pd.DataFrame) -> pd.DataFrame:
    """Reserved ablation table shape (empty rows until ablations are configured)."""
    columns = ["dataset", "comparison", "delta_f1_macro", "delta_dp_diff"]
    return pd.DataFrame(columns=columns)


def _fit_method_estimator(spec, X_fit, y_fit, sensitive_train=None):
    if spec.use_fair_reweight and sensitive_train is not None:
        weights = compute_fair_reweighting(y_fit, sensitive_train)
        spec.estimator.fit(X_fit, y_fit, sample_weight=weights)
        return
    # Pass sensitive when estimator opts into fairness-aware fit (e.g. HIES).
    if sensitive_train is not None and hasattr(spec.estimator, "fairness_threshold_penalty"):
        try:
            spec.estimator.fit(X_fit, y_fit, sensitive=sensitive_train)
            return
        except TypeError:
            pass
    spec.estimator.fit(X_fit, y_fit)


def _predict_with_threshold_mode(spec, y_prob: np.ndarray, sensitive: np.ndarray, default_threshold: float, group_thresholds):
    if spec.use_group_thresholds and isinstance(group_thresholds, dict) and group_thresholds:
        sensitive = np.asarray(sensitive).astype(str)
        y_pred = np.zeros(len(y_prob), dtype=int)
        for group in np.unique(sensitive):
            mask = sensitive == group
            threshold = float(group_thresholds.get(group, default_threshold))
            y_pred[mask] = (y_prob[mask] >= threshold).astype(int)
        return y_pred
    return (y_prob >= default_threshold).astype(int)


def _build_tradeoff_table(raw: pd.DataFrame, opt_cfg: dict) -> pd.DataFrame:
    rows: list[dict] = []
    baseline_name = str(opt_cfg.get("tradeoff_baseline_method", "xgboost"))
    target_name = str(opt_cfg.get("tradeoff_target_method", "calibrated_xgboost"))
    for dataset, sub in raw.groupby("dataset"):
        base = sub[sub["method"] == baseline_name].copy()
        targ = sub[sub["method"] == target_name].copy()
        if base.empty or targ.empty:
            continue
        merged = targ.merge(
            base,
            on=["dataset", "seed", "fold"],
            suffixes=("_target", "_baseline"),
            how="inner",
        )
        if merged.empty:
            continue
        delta_f1 = merged["f1_macro_target"] - merged["f1_macro_baseline"]
        delta_dp = merged["demographic_parity_diff_target"] - merged["demographic_parity_diff_baseline"]
        delta_eod = merged["equal_opportunity_diff_target"] - merged["equal_opportunity_diff_baseline"]
        fairness_improved = (delta_dp < 0).astype(float)
        joint_wins = ((delta_dp < 0) & (delta_f1 >= 0)).astype(float)
        rows.append(
            {
                "dataset": dataset,
                "baseline_method": baseline_name,
                "target_method": target_name,
                "paired_samples": int(len(merged)),
                "delta_f1_macro_mean": float(delta_f1.mean()),
                "delta_f1_macro_std": float(delta_f1.std(ddof=1)) if len(delta_f1) > 1 else 0.0,
                "delta_dp_diff_mean": float(delta_dp.mean()),
                "delta_eod_diff_mean": float(delta_eod.mean()),
                "fairness_win_rate": float(fairness_improved.mean()),
                "joint_win_rate": float(joint_wins.mean()),
            }
        )
    columns = [
        "dataset",
        "baseline_method",
        "target_method",
        "paired_samples",
        "delta_f1_macro_mean",
        "delta_f1_macro_std",
        "delta_dp_diff_mean",
        "delta_eod_diff_mean",
        "fairness_win_rate",
        "joint_win_rate",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(["dataset"])


def _build_guardrail_table(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["dataset", "method"], as_index=False)
        .agg(
            guardrail_pass_rate=("fairness_guardrail_pass", "mean"),
            dp_diff_mean=("demographic_parity_diff", "mean"),
            eod_diff_mean=("equal_opportunity_diff", "mean"),
        )
        .sort_values(["dataset", "guardrail_pass_rate", "method"], ascending=[True, False, True])
    )


def run_benchmark(
    config,
    run_id: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if run_id is None:
        run_id = "adhoc"
    _setup_mlflow(config, run_id=run_id)
    if logger:
        logger.info("MLflow run started with run_id=%s", run_id)
    rows: list[dict] = []
    opt_cfg = _resolve_optimization_config(config)
    data_cfg = config.data
    for dataset_name in data_cfg["dataset_order"]:
        if logger:
            logger.info("Loading dataset=%s", dataset_name)
        bundle = load_dataset_bundle(
            dataset_name,
            data_cfg["raw_dir"],
            seed=config.seed,
            allow_synthetic=data_cfg.get("use_synthetic_if_missing", True),
        )
        X, y, sensitive = _prepare_xy(
            bundle.frame,
            bundle.target_col,
            bundle.sensitive_col,
            leakage_columns=bundle.leakage_columns,
            enable_feature_interactions=bool(data_cfg.get("enable_feature_interactions", False)),
            max_interactions=int(data_cfg.get("max_feature_interactions", 0)),
        )
        class_ratio = max(y.sum(), 1) / max((len(y) - y.sum()), 1)
        class_weight_ratio = max((len(y) - y.sum()), 1) / max(y.sum(), 1)
        for seed in config.seeds:
            if logger:
                logger.info("Dataset=%s | seed=%s", dataset_name, seed)
            cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=seed)
            for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
                X_train = X.iloc[tr_idx].copy()
                y_train = y.iloc[tr_idx].copy()
                s_train = sensitive.iloc[tr_idx].copy()
                X_test = X.iloc[te_idx].copy()
                y_test = y.iloc[te_idx].copy()
                s_test = sensitive.iloc[te_idx].copy()
                pre = make_preprocessor(X_train)
                X_train_t = pre.fit_transform(X_train)
                X_test_t = pre.transform(X_test)
                if hasattr(X_train_t, "toarray"):
                    X_train_t = X_train_t.toarray()
                if hasattr(X_test_t, "toarray"):
                    X_test_t = X_test_t.toarray()
                inner_split = None
                baseline_inner_ref = None
                if len(np.unique(y_train.values)) > 1 and len(y_train) >= 20:
                    tr_idx_inner, val_idx_inner = train_test_split(
                        np.arange(len(y_train)),
                        test_size=float(opt_cfg["threshold_validation_fraction"]),
                        stratify=y_train.values,
                        random_state=seed * 100 + fold,
                    )
                    inner_split = (tr_idx_inner, val_idx_inner)
                    baseline_spec_inner = build_method(
                        "xgboost",
                        seed=seed,
                        class_weight_ratio=class_weight_ratio,
                        method_params=opt_cfg["method_params"],
                    )
                    X_tr_inner = X_train_t[tr_idx_inner]
                    y_tr_inner = y_train.iloc[tr_idx_inner]
                    X_fit_base, y_fit_base = _apply_method_resampling(
                        baseline_spec_inner,
                        X_tr_inner,
                        y_tr_inner,
                    )
                    _fit_method_estimator(
                        baseline_spec_inner,
                        X_fit_base,
                        y_fit_base,
                        s_train.iloc[tr_idx_inner].values,
                    )
                    X_val_inner = X_train_t[val_idx_inner]
                    y_val_inner = y_train.iloc[val_idx_inner]
                    s_val_inner = s_train.iloc[val_idx_inner]
                    y_prob_base = baseline_spec_inner.estimator.predict_proba(X_val_inner)[:, 1]
                    y_pred_base = (y_prob_base >= 0.5).astype(int)
                    baseline_inner_ref = compute_fairness(y_val_inner.values, y_pred_base, s_val_inner.values)
                for method_name in config.methods:
                    if logger:
                        logger.info("Dataset=%s | seed=%s | fold=%s | method=%s", dataset_name, seed, fold, method_name)
                    spec = build_method(
                        method_name,
                        seed=seed,
                        class_weight_ratio=class_weight_ratio,
                        method_params=opt_cfg["method_params"],
                    )
                    threshold = 0.5
                    group_thresholds = None
                    method_opt_cfg = deepcopy(opt_cfg)
                    if (
                        opt_cfg["enable_threshold_optimization"]
                        and method_name in opt_cfg["threshold_methods"]
                        and inner_split is not None
                    ):
                        tr_idx_inner, val_idx_inner = inner_split
                        X_tr_inner = X_train_t[tr_idx_inner]
                        y_tr_inner = y_train.iloc[tr_idx_inner]
                        s_tr_inner = s_train.iloc[tr_idx_inner]
                        X_val_inner = X_train_t[val_idx_inner]
                        y_val_inner = y_train.iloc[val_idx_inner]
                        s_val_inner = s_train.iloc[val_idx_inner]
                        spec_tune = build_method(
                            method_name,
                            seed=seed,
                            class_weight_ratio=class_weight_ratio,
                            method_params=opt_cfg["method_params"],
                        )
                        X_fit_inner, y_fit_inner = _apply_method_resampling(
                            spec_tune,
                            X_tr_inner,
                            y_tr_inner,
                        )
                        _fit_method_estimator(spec_tune, X_fit_inner, y_fit_inner, s_tr_inner.values)
                        y_prob_val = spec_tune.estimator.predict_proba(X_val_inner)[:, 1]
                        if spec_tune.use_group_thresholds:

                            def _score_fn(y_t, y_p, y_pb, sens):
                                perf_s = compute_performance(y_t, y_p, y_pb)
                                fair_s = compute_fairness(y_t, y_p, sens)
                                return _compute_joint_score(
                                    perf=perf_s,
                                    fair=fair_s,
                                    weights=dict(method_opt_cfg["objective_weights"]),
                                )

                            group_thresholds = find_group_thresholds(
                                y_true=y_val_inner.values,
                                y_prob=y_prob_val,
                                sensitive=s_val_inner.values,
                                threshold_grid=list(opt_cfg["threshold_grid"]),
                                score_fn=_score_fn,
                                baseline_dp=(
                                    baseline_inner_ref["demographic_parity_diff"]
                                    if baseline_inner_ref is not None
                                    else None
                                ),
                                baseline_eod=(
                                    baseline_inner_ref["equal_opportunity_diff"]
                                    if baseline_inner_ref is not None
                                    else None
                                ),
                                guardrail_fn=compute_fairness,
                                max_dp_increase=float(opt_cfg["max_dp_increase"]),
                                max_eod_increase=float(opt_cfg["max_eod_increase"]),
                            )
                        else:
                            threshold = _find_best_threshold(
                                y_true=y_val_inner.values,
                                y_prob=y_prob_val,
                                sensitive=s_val_inner.values,
                                threshold_grid=list(opt_cfg["threshold_grid"]),
                                weights=dict(method_opt_cfg["objective_weights"]),
                                baseline_dp=(
                                    baseline_inner_ref["demographic_parity_diff"]
                                    if baseline_inner_ref is not None
                                    else None
                                ),
                                baseline_eod=(
                                    baseline_inner_ref["equal_opportunity_diff"]
                                    if baseline_inner_ref is not None
                                    else None
                                ),
                                max_dp_increase=float(opt_cfg["max_dp_increase"]),
                                max_eod_increase=float(opt_cfg["max_eod_increase"]),
                            )
                    X_fit, y_fit = _apply_method_resampling(
                        spec,
                        X_train_t,
                        y_train,
                    )
                    t_fit_start = time.perf_counter()
                    _fit_method_estimator(spec, X_fit, y_fit, s_train.values)
                    t_fit = float(time.perf_counter() - t_fit_start)

                    # Adopt estimator.threshold_ when runner did not tune (HIES OOF).
                    if (
                        method_name not in opt_cfg["threshold_methods"]
                        and not getattr(spec, "use_group_thresholds", False)
                        and not getattr(spec, "use_fairlearn_equalized_odds", False)
                    ):
                        internal_thr = getattr(spec.estimator, "threshold_", None)
                        if internal_thr is not None:
                            threshold = float(internal_thr)

                    fairlearn_optimizer = None
                    if getattr(spec, "use_fairlearn_equalized_odds", False):
                        try:
                            if inner_split is not None:
                                va_ix = inner_split[1]
                                sens_va = s_train.iloc[va_ix].astype(str).to_numpy()
                                if len(np.unique(sens_va)) >= 2:
                                    from fairlearn.postprocessing import ThresholdOptimizer

                                    fairlearn_optimizer = ThresholdOptimizer(
                                        estimator=spec.estimator,
                                        prefit=True,
                                        constraints="equalized_odds",
                                    )
                                    fairlearn_optimizer.fit(
                                        X_train_t[va_ix],
                                        y_train.iloc[va_ix].values,
                                        sensitive_features=sens_va,
                                    )
                        except (ValueError, RuntimeError):
                            fairlearn_optimizer = None

                    t_pred_start = time.perf_counter()
                    if fairlearn_optimizer is not None:
                        y_prob = np.asarray(spec.estimator.predict_proba(X_test_t)[:, 1], dtype=np.float64)
                        y_pred = fairlearn_optimizer.predict(
                            X_test_t,
                            sensitive_features=s_test.astype(str).to_numpy(),
                        )
                        if y_pred.ndim > 1:
                            y_pred = y_pred.ravel()
                        y_pred = y_pred.astype(int)
                        threshold = -1.0
                    else:
                        y_prob = spec.estimator.predict_proba(X_test_t)[:, 1]
                        y_pred = _predict_with_threshold_mode(
                            spec, y_prob, s_test.values, threshold, group_thresholds
                        )
                    t_pred = float(time.perf_counter() - t_pred_start)
                    perf = compute_performance(y_test.values, y_pred, y_prob)
                    fair = compute_fairness(y_test.values, y_pred, s_test.values)
                    cal = compute_calibration(y_test.values, y_prob)
                    cal_par = compute_calibration_per_group(
                        y_test.values, y_prob, s_test.values
                    )
                    sub = {}
                    s_arr = s_test.astype(str).to_numpy()
                    y_te_arr = y_test.values
                    for grp in np.unique(s_arr):
                        mask = s_arr == grp
                        if mask.sum() < 5:
                            continue
                        from sklearn.metrics import f1_score as _f1
                        from sklearn.metrics import recall_score as _rec
                        from sklearn.metrics import precision_score as _prec
                        y_t_g = y_te_arr[mask]
                        y_p_g = y_pred[mask]
                        sub[f"f1_macro_grp_{grp}"] = float(
                            _f1(y_t_g, y_p_g, average="macro")
                        )
                        sub[f"tpr_grp_{grp}"] = float(
                            _rec(y_t_g, y_p_g, zero_division=0)
                        )
                        neg = y_t_g == 0
                        if int(neg.sum()) >= 1:
                            sub[f"fpr_grp_{grp}"] = float(y_p_g[neg].mean())
                        else:
                            sub[f"fpr_grp_{grp}"] = 0.0
                        sub[f"ppv_grp_{grp}"] = float(
                            _prec(y_t_g, y_p_g, zero_division=0)
                        )
                    row = {
                        "dataset": dataset_name,
                        "seed": seed,
                        "fold": fold,
                        "method": method_name,
                        "class_ratio": class_ratio,
                        "decision_threshold": threshold,
                        "group_thresholds": "" if not group_thresholds else str(group_thresholds),
                        "n_test": int(len(y_test)),
                        "n_train": int(len(y_train)),
                        "fit_time_sec": t_fit,
                        "predict_time_sec": t_pred,
                    }
                    row.update(perf)
                    row.update(fair)
                    row.update(cal)
                    row.update(cal_par)
                    row.update(sub)
                    row["joint_score"] = _compute_joint_score(
                        perf=perf,
                        fair=fair,
                        weights=dict(method_opt_cfg["objective_weights"]),
                    )
                    rows.append(row)
                    if mlflow.active_run() is not None:
                        _log_mlflow_fold_metrics(row)
    raw = pd.DataFrame(rows)
    raw = _apply_fairness_guardrails(
        raw,
        max_dp_increase=float(opt_cfg["max_dp_increase"]),
        max_eod_increase=float(opt_cfg["max_eod_increase"]),
    )
    summary = (
        raw.groupby(["dataset", "method"], as_index=False)
        .agg(
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            mcc_mean=("mcc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            bal_acc_mean=("balanced_accuracy", "mean"),
            dp_diff_mean=("demographic_parity_diff", "mean"),
            eod_diff_mean=("equal_opportunity_diff", "mean"),
            decision_threshold_mean=("decision_threshold", "mean"),
            joint_score_mean=("joint_score", "mean"),
            fairness_guardrail_pass_rate=("fairness_guardrail_pass", "mean"),
        )
        .sort_values(["dataset", "f1_macro_mean"], ascending=[True, False])
    )
    ci_rows: list[dict[str, float | str]] = []
    for (dataset_name, method_name), sub in raw.groupby(["dataset", "method"], sort=False):
        arr = sub["f1_macro"].to_numpy()
        if len(arr) > 1:
            res = bootstrap((arr,), np.mean, n_resamples=1000, random_state=0, method="basic")
            low = float(res.confidence_interval.low)
            high = float(res.confidence_interval.high)
        else:
            low = float(arr[0])
            high = float(arr[0])
        ci_rows.append(
            {
                "dataset": dataset_name,
                "method": method_name,
                "f1_macro_ci_low": low,
                "f1_macro_ci_high": high,
            }
        )
    ci_frame = pd.DataFrame(ci_rows)
    summary = summary.merge(ci_frame, on=["dataset", "method"], how="left")
    if mlflow.active_run() is not None:
        for _, row in summary.iterrows():
            base = f"{row['dataset']}/{row['method']}/summary"
            mlflow.log_metric(f"{base}/f1_macro_mean", float(row["f1_macro_mean"]))
            mlflow.log_metric(f"{base}/mcc_mean", float(row["mcc_mean"]))
            mlflow.log_metric(f"{base}/pr_auc_mean", float(row["pr_auc_mean"]))
            mlflow.log_metric(f"{base}/bal_acc_mean", float(row["bal_acc_mean"]))
            mlflow.log_metric(f"{base}/dp_diff_mean", float(row["dp_diff_mean"]))
            mlflow.log_metric(f"{base}/eod_diff_mean", float(row["eod_diff_mean"]))
    return raw, summary


def finalize_mlflow(logger: logging.Logger | None = None) -> None:
    if mlflow.active_run() is not None:
        mlflow.end_run()
        if logger:
            logger.info("MLflow run closed.")


def save_results(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
    run_id: str,
    logger: logging.Logger | None = None,
    optimization: dict | None = None,
) -> None:
    out = Path(output_dir)
    results_dir = out / "results"
    tables_dir = out / "tables"
    manifests_dir = out / "manifests"
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(results_dir / "results_raw.csv", index=False)
    summary.to_csv(results_dir / "results_summary.csv", index=False)
    summary.to_csv(tables_dir / "table_main_results.csv", index=False)
    fairness_table = _build_fairness_table(raw)
    fairness_table.to_csv(tables_dir / "table_fairness_results.csv", index=False)
    significance_table = _build_significance_table(raw)
    significance_table.to_csv(tables_dir / "table_significance.csv", index=False)
    ablation_table = _build_ablation_table(summary)
    ablation_table.to_csv(tables_dir / "table_ablation.csv", index=False)
    opt_cfg = optimization if isinstance(optimization, dict) else {}
    tradeoff_table = _build_tradeoff_table(raw, opt_cfg)
    tradeoff_table.to_csv(tables_dir / "table_tradeoff.csv", index=False)
    guardrail_table = _build_guardrail_table(raw)
    guardrail_table.to_csv(tables_dir / "table_guardrail.csv", index=False)
    best_per_ds = select_best_per_dataset(summary)
    best_per_ds.to_csv(tables_dir / "table_best_per_dataset.csv", index=False)
    table_manifest = pd.DataFrame(
        [
            {
                "table_id": "table_main_results",
                "purpose": "Primary performance comparison per (dataset, method).",
                "source_csv": "artifacts/results/results_summary.csv",
                "key_columns": "dataset,method,f1_macro_mean,mcc_mean",
            },
            {
                "table_id": "table_fairness_results",
                "purpose": "Per-(dataset, method) demographic-parity and equal-opportunity gaps.",
                "source_csv": "artifacts/results/results_raw.csv",
                "key_columns": "dataset,method,dp_diff_mean,eod_diff_mean",
            },
            {
                "table_id": "table_ablation",
                "purpose": "Reserved ablation slot (populated by external ablation scripts).",
                "source_csv": "artifacts/results/results_summary.csv",
                "key_columns": "dataset,comparison,delta_f1_macro,delta_dp_diff",
            },
            {
                "table_id": "table_significance",
                "purpose": "Paired Wilcoxon vs xgboost baseline with Holm and BH corrections.",
                "source_csv": "artifacts/results/results_raw.csv",
                "key_columns": "dataset,method,mean_delta_f1_macro,p_value",
            },
            {
                "table_id": "table_tradeoff",
                "purpose": "Per-(seed, fold) paired delta in F1, DP, EoD between target and baseline.",
                "source_csv": "artifacts/results/results_raw.csv",
                "key_columns": "dataset,paired_samples,delta_f1_macro_mean,delta_dp_diff_mean,joint_win_rate",
            },
            {
                "table_id": "table_guardrail",
                "purpose": "Per-method pass rate for the fairness guardrail relative to the xgboost baseline.",
                "source_csv": "artifacts/results/results_raw.csv",
                "key_columns": "dataset,method,guardrail_pass_rate,dp_diff_mean,eod_diff_mean",
            },
            {
                "table_id": "table_best_per_dataset",
                "purpose": "Best method per dataset by macro-F1, tie-broken on joint_score / MCC / PR-AUC.",
                "source_csv": "artifacts/results/results_summary.csv",
                "key_columns": "dataset,method,f1_macro_mean",
            },
        ]
    )
    table_manifest.to_csv(manifests_dir / "table_manifest.csv", index=False)
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(results_dir / "results_raw.csv"))
        mlflow.log_artifact(str(results_dir / "results_summary.csv"))
        mlflow.log_artifact(str(tables_dir / "table_main_results.csv"))
        mlflow.log_artifact(str(tables_dir / "table_fairness_results.csv"))
        mlflow.log_artifact(str(tables_dir / "table_significance.csv"))
        mlflow.log_artifact(str(tables_dir / "table_ablation.csv"))
        mlflow.log_artifact(str(tables_dir / "table_tradeoff.csv"))
        mlflow.log_artifact(str(tables_dir / "table_guardrail.csv"))
        mlflow.log_artifact(str(tables_dir / "table_best_per_dataset.csv"))
        mlflow.log_artifact(str(manifests_dir / "table_manifest.csv"))
    if logger:
        logger.info("Core results and tables written for run_id=%s", run_id)
