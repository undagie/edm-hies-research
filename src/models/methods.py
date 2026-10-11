"""Method factory and HIES estimator for the curated benchmark.

``calibrated_xgboost`` is intentionally excluded from the default config to
avoid double-counting: HIES already uses a sigmoid-Platt calibrated XGBoost
as one of its five bases. The factory entry remains for ablation use.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

try:
    from tabpfn import TabPFNClassifier
except ImportError:  # pragma: no cover
    TabPFNClassifier = None


class _AutoCalibratedEstimator(BaseEstimator, ClassifierMixin):
    """Wraps a learner in ``CalibratedClassifierCV`` choosing method at fit-time.

    If ``calibration_method`` is ``"auto"``, uses ``"isotonic"`` when training
    set size reaches ``n_train_threshold``, otherwise ``\"sigmoid\"``.
    """

    def __init__(
        self,
        estimator,
        *,
        cv: int = 3,
        calibration_method: str = "auto",
        n_train_threshold: int = 1500,
    ):
        self.estimator = estimator
        self.cv = cv
        self.calibration_method = calibration_method
        self.n_train_threshold = n_train_threshold

    def fit(self, X, y, sample_weight=None):  # noqa: ANN001
        method = self.calibration_method
        if method == "auto":
            n = len(np.asarray(y))
            method = "isotonic" if n >= self.n_train_threshold else "sigmoid"
        self.method_ = method
        inner = clone(self.estimator)
        cal = CalibratedClassifierCV(estimator=inner, method=method, cv=self.cv)
        if sample_weight is not None:
            cal.fit(X, y, sample_weight=sample_weight)
        else:
            cal.fit(X, y)
        self.calibrated_ = cal
        self.classes_ = cal.classes_
        return self

    def predict(self, X):
        return self.calibrated_.predict(X)

    def predict_proba(self, X):
        return np.asarray(self.calibrated_.predict_proba(X), dtype=np.float64)


class _PredictProbaFloat64Wrapper(BaseEstimator, ClassifierMixin):
    """Cast predict_proba to float64 for fairlearn ThresholdOptimizer.

    Fairlearn writes into float64 Series; XGBoost emits float32 and triggers
    FutureWarning. Casting here silences that without changing numerics.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator
        self._estimator = estimator

    def fit(self, X, y, sample_weight=None):  # noqa: ANN001
        if sample_weight is not None:
            self.estimator.fit(X, y, sample_weight=sample_weight)
        else:
            self.estimator.fit(X, y)
        self._estimator = self.estimator
        if hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return np.asarray(self.estimator.predict_proba(X), dtype=np.float64)


@dataclass
class MethodSpec:
    name: str
    estimator: object
    resampler: object | None
    use_fair_reweight: bool = False
    use_group_thresholds: bool = False
    use_fairlearn_equalized_odds: bool = False


def _calibrated_xgb(seed: int, class_weight_ratio: float, model_params: dict | None = None) -> CalibratedClassifierCV:
    inner = _base_xgb(seed, class_weight=class_weight_ratio, model_params=model_params)
    return CalibratedClassifierCV(estimator=inner, method="sigmoid", cv=3)


def _base_lgbm(seed: int, class_weight_ratio: float | None = None, model_params: dict | None = None):
    if LGBMClassifier is None:
        raise ImportError("lightgbm is required for method `lightgbm`; install with pip install lightgbm")
    params = {
        "n_estimators": 260,
        "max_depth": 6,
        "learning_rate": 0.04,
        "reg_lambda": 1.2,
        "random_state": seed,
        "verbose": -1,
        "force_col_wise": True,
        "n_jobs": -1,
    }
    if class_weight_ratio is not None:
        params["scale_pos_weight"] = float(class_weight_ratio)
    if model_params:
        params.update(model_params)
    return LGBMClassifier(**params)


def _base_lr(seed: int, model_params: dict | None = None) -> LogisticRegression:
    params = {
        "max_iter": 2000,
        "random_state": seed,
        "class_weight": "balanced",
        "solver": "lbfgs",
    }
    if model_params:
        params.update(model_params)
    return LogisticRegression(**params)


def _base_catboost(seed: int, class_weight_ratio: float | None = None, model_params: dict | None = None):
    if CatBoostClassifier is None:
        raise ImportError("catboost is required for method `catboost`; install with pip install catboost")
    params = {
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    if class_weight_ratio is not None:
        # Mirror scale_pos_weight: class_weights = [w0, w1] with w1 = neg/pos.
        params["class_weights"] = [1.0, float(class_weight_ratio)]
    if model_params:
        params.update(model_params)
    return CatBoostClassifier(**params)


class _TabPFNAdapter(BaseEstimator, ClassifierMixin):
    """Sklearn-style TabPFN adapter with stratified train cap and CUDA/CPU device."""

    def __init__(self, *, seed: int = 0, max_n_train: int = 2000, device: str = "auto"):
        self.seed = seed
        self.max_n_train = max_n_train
        self.device = device

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def fit(self, X, y, sample_weight=None):  # noqa: ANN001
        if TabPFNClassifier is None:
            raise ImportError("tabpfn is required for method `tabpfn`; install with pip install tabpfn")
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y).astype(int)
        n = len(y_arr)
        rng = np.random.default_rng(self.seed)
        if n > self.max_n_train:
            classes, counts = np.unique(y_arr, return_counts=True)
            idx_keep = []
            for cls, _ in zip(classes, counts, strict=False):
                cls_idx = np.where(y_arr == cls)[0]
                share = int(round(len(cls_idx) * (self.max_n_train / n)))
                share = max(1, share)
                pick = rng.choice(cls_idx, size=min(share, len(cls_idx)), replace=False)
                idx_keep.append(pick)
            keep = np.sort(np.concatenate(idx_keep))
            X_arr = X_arr[keep]
            y_arr = y_arr[keep]
        device = self._resolve_device()
        try:
            self.estimator_ = TabPFNClassifier(random_state=self.seed, device=device)
        except TypeError:
            # Older TabPFN builds lack the device kwarg.
            self.estimator_ = TabPFNClassifier(random_state=self.seed)
        self.estimator_.fit(X_arr, y_arr)
        self.classes_ = np.asarray(getattr(self.estimator_, "classes_", np.unique(y_arr)))
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        return np.asarray(self.estimator_.predict_proba(X_arr), dtype=np.float64)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def _base_xgb(seed: int, class_weight: float | None = None, model_params: dict | None = None) -> XGBClassifier:
    kwargs = {
        "n_estimators": 260,
        "max_depth": 4,
        "min_child_weight": 2,
        "gamma": 0.1,
        "learning_rate": 0.04,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.2,
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": -1,
    }
    if class_weight is not None:
        kwargs["scale_pos_weight"] = class_weight
    if model_params:
        kwargs.update(model_params)
    return XGBClassifier(**kwargs)


def _heterogeneous_calibrated_bases(
    seed: int,
    class_weight_ratio: float,
    xgb_params: dict | None = None,
    rf_params: dict | None = None,
    hgb_params: dict | None = None,
    lgbm_params: dict | None = None,
    cat_params: dict | None = None,
    include_catboost: bool = True,
    calibration_cv: int = 3,
    calibration_method: str = "sigmoid",
    n_train_threshold: int = 1500,
) -> list[tuple[str, object]]:
    """Imbalance-adaptive learners wrapped with auto or fixed calibration.

    Optionally includes CatBoost as a fifth base when installed.
    """

    wrapped_method = calibration_method
    xgb = _base_xgb(seed, class_weight=class_weight_ratio, model_params=xgb_params)
    rf_merged = {
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_leaf": 1,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": seed,
    }
    rf_merged.update(rf_params or {})
    rf = RandomForestClassifier(**rf_merged)
    hgb_merged = {
        "max_iter": 320,
        "max_depth": 6,
        "learning_rate": 0.05,
        "class_weight": "balanced",
        "random_state": seed,
        "early_stopping": False,
    }
    hgb_merged.update(hgb_params or {})
    hgb = HistGradientBoostingClassifier(**hgb_merged)
    if LGBMClassifier is None:
        raise ImportError("lightgbm is required for HIES ensembles; install lightgbm.")
    lgbm_merged = {
        "n_estimators": 320,
        "max_depth": -1,
        "num_leaves": 63,
        "learning_rate": 0.04,
        "reg_lambda": 1.2,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
        "force_col_wise": True,
        "scale_pos_weight": float(class_weight_ratio),
    }
    lgbm_merged.update(lgbm_params or {})
    lgbm = LGBMClassifier(**lgbm_merged)
    bases_raw: list[tuple[str, object]] = [
        ("xgb_spw", xgb),
        ("rf_balanced", rf),
        ("hgb_balanced", hgb),
        ("lgbm_spw", lgbm),
    ]
    if include_catboost and CatBoostClassifier is not None:
        # auto_class_weights="Balanced" is clone-safe (literal class_weights breaks clone).
        cat_merged = {
            "iterations": 280,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
            "auto_class_weights": "Balanced",
        }
        cat_merged.update(cat_params or {})
        bases_raw.append(("cat_balanced", CatBoostClassifier(**cat_merged)))
    return [
        (
            name,
            _AutoCalibratedEstimator(
                est,
                cv=calibration_cv,
                calibration_method=wrapped_method,
                n_train_threshold=n_train_threshold,
            ),
        )
        for name, est in bases_raw
    ]


def _margin_from_proba(proba: np.ndarray) -> np.ndarray:
    """Return confidence margin (top1 - top2) per sample."""
    arr = np.asarray(proba, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] <= 1:
        return np.ones(arr.shape[0], dtype=np.float64)
    top2 = np.partition(arr, kth=arr.shape[1] - 2, axis=1)[:, -2:]
    return np.clip(top2[:, 1] - top2[:, 0], 0.0, 1.0)


def _joint_perf_score_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    f1_weight: float,
    mcc_weight: float,
) -> float:
    """Utility-only joint score for threshold selection in binary settings."""
    f1_val = float(f1_score(y_true, y_pred, average="macro"))
    mcc_val = float(matthews_corrcoef(y_true, y_pred))
    return f1_weight * f1_val + mcc_weight * mcc_val


def _fairness_gap_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> tuple[float, float]:
    """Return (dp_diff, eod_diff) over the provided sensitive vector.

    Both metrics are max-over-groups minus min-over-groups; returns ``(0, 0)``
    when fewer than two groups are observed.
    """
    s = np.asarray(sensitive)
    groups = np.unique(s)
    if groups.size < 2:
        return 0.0, 0.0
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    sel_rates: list[float] = []
    tpr_rates: list[float] = []
    for g in groups:
        mask = s == g
        if not mask.any():
            continue
        sel_rates.append(float(y_p[mask].mean()))
        pos = mask & (y_t == 1)
        if pos.any():
            tpr_rates.append(float(y_p[pos].mean()))
    if len(sel_rates) < 2:
        return 0.0, 0.0
    dp = float(max(sel_rates) - min(sel_rates))
    eod = float(max(tpr_rates) - min(tpr_rates)) if len(tpr_rates) >= 2 else 0.0
    return dp, eod


class _HIESAdaptive(BaseEstimator, ClassifierMixin):
    """HIES: OOF-pruned, per-sample confidence-weighted calibrated ensemble.

    With ``auto_config``, knobs resolve at fit-time from n_train / imbalance /
    feature-dim buckets (no dataset-name metadata). With ``fusion="auto"``,
    chooses weighted vs stacked fusion via fold-only OOF macro-F1 (no test leakage).
    """

    def __init__(
        self,
        estimators,
        *,
        prune_delta: float = 0.025,
        min_keep: int = 3,
        temperature: float = 0.08,
        margin_floor: float = 0.05,
        cv_splits: int = 5,
        tune_threshold: bool = True,
        threshold_grid: list[float] | None = None,
        threshold_clip: tuple[float, float] = (0.30, 0.70),
        objective_weights: dict[str, float] | None = None,
        auto_config: bool = False,
        auto_n_large: int = 2000,
        auto_imb_thresh: float = 2.5,
        auto_small: dict | None = None,
        auto_large: dict | None = None,
        auto_feature_dim_split: int | None = None,
        auto_wide: dict | None = None,
        auto_narrow: dict | None = None,
        auto_imbalanced_mcc_boost: float = 0.0,
        fusion: str = "weighted",
        stack_meta_C: float = 1.0,
        stack_meta_cv: int = 5,
        stack_select_margin: float = 0.0,
        recalibrate_fusion: bool = False,
        recalibration_method: str = "auto",
        recalibration_cv: int = 5,
        fairness_threshold_penalty: float = 0.0,
        random_state: int | None = None,
    ):
        self.estimators = estimators
        self.prune_delta = prune_delta
        self.min_keep = min_keep
        self.temperature = temperature
        self.margin_floor = margin_floor
        self.cv_splits = cv_splits
        self.tune_threshold = tune_threshold
        self.threshold_grid = threshold_grid
        self.threshold_clip = threshold_clip
        self.objective_weights = objective_weights
        self.auto_config = auto_config
        self.auto_n_large = auto_n_large
        self.auto_imb_thresh = auto_imb_thresh
        self.auto_small = auto_small
        self.auto_large = auto_large
        self.auto_feature_dim_split = auto_feature_dim_split
        self.auto_wide = auto_wide
        self.auto_narrow = auto_narrow
        self.auto_imbalanced_mcc_boost = auto_imbalanced_mcc_boost
        self.fusion = fusion
        self.stack_meta_C = stack_meta_C
        self.stack_meta_cv = stack_meta_cv
        self.stack_select_margin = stack_select_margin
        self.recalibrate_fusion = recalibrate_fusion
        self.recalibration_method = recalibration_method
        self.recalibration_cv = recalibration_cv
        self.fairness_threshold_penalty = fairness_threshold_penalty
        self.random_state = random_state

    def _per_sample_weights(self, margins: np.ndarray, global_w: np.ndarray) -> np.ndarray:
        # Margin floor prevents low-confidence bases from collapsing to zero weight.
        floor = float(self.margin_floor)
        scores = (margins + floor) * global_w.reshape(1, -1)
        denom = np.clip(scores.sum(axis=1, keepdims=True), 1e-12, None)
        return scores / denom

    def _resolve_effective_config(
        self,
        n_train: int,
        class_ratio: float,
        n_features: int | None = None,
    ) -> dict:
        obj = dict(self.objective_weights or {})
        defaults = {
            "prune_delta": float(self.prune_delta),
            "temperature": float(self.temperature),
            "f1_weight": float(obj.get("f1_macro", 0.88)),
            "mcc_weight": float(obj.get("mcc", 0.50)),
            "threshold_grid": list(self.threshold_grid or [0.40, 0.43, 0.45, 0.47, 0.50, 0.53, 0.55]),
            "fusion": str(self.fusion or "weighted").lower(),
        }
        if not self.auto_config:
            return defaults
        bucket = "large" if int(n_train) >= int(self.auto_n_large) else "small"
        preset = (self.auto_large if bucket == "large" else self.auto_small) or {}
        out = {
            "prune_delta": float(preset.get("prune_delta", defaults["prune_delta"])),
            "temperature": float(preset.get("temperature", defaults["temperature"])),
            "f1_weight": float(preset.get("f1_weight", defaults["f1_weight"])),
            "mcc_weight": float(preset.get("mcc_weight", defaults["mcc_weight"])),
            "threshold_grid": list(preset.get("threshold_grid", defaults["threshold_grid"])),
            "fusion": str(preset.get("fusion", defaults["fusion"])).lower(),
        }
        split_thr = self.auto_feature_dim_split
        if split_thr is not None and int(split_thr) > 0 and n_features is not None:
            dim_preset = (self.auto_wide if int(n_features) >= int(split_thr) else self.auto_narrow) or {}
            if dim_preset:
                if "prune_delta" in dim_preset:
                    out["prune_delta"] = float(dim_preset["prune_delta"])
                if "temperature" in dim_preset:
                    out["temperature"] = float(dim_preset["temperature"])
                if "f1_weight" in dim_preset:
                    out["f1_weight"] = float(dim_preset["f1_weight"])
                if "mcc_weight" in dim_preset:
                    out["mcc_weight"] = float(dim_preset["mcc_weight"])
                if "threshold_grid" in dim_preset:
                    out["threshold_grid"] = list(dim_preset["threshold_grid"])
                if "fusion" in dim_preset:
                    out["fusion"] = str(dim_preset["fusion"]).lower()
        if float(class_ratio) >= float(self.auto_imb_thresh):
            out["mcc_weight"] = out["mcc_weight"] + float(self.auto_imbalanced_mcc_boost)
        return out

    def fit(self, X, y, sample_weight=None, sensitive=None):  # noqa: ANN001
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        n_min = int(np.bincount(y_arr).min()) if y_arr.size else 0
        n_max = int(np.bincount(y_arr).max()) if y_arr.size else 1
        n_train = int(len(y_arr))
        class_ratio = float(n_max) / float(max(n_min, 1))
        n_features = int(np.asarray(X).shape[1])
        eff = self._resolve_effective_config(
            n_train=n_train,
            class_ratio=class_ratio,
            n_features=n_features,
        )
        self.effective_config_ = eff
        sensitive_arr = None
        if sensitive is not None:
            sensitive_arr = np.asarray(sensitive)
            if sensitive_arr.shape[0] != y_arr.shape[0]:
                sensitive_arr = None
        self.sensitive_seen_in_fit_ = sensitive_arr is not None

        n_splits = max(2, min(int(self.cv_splits), n_min))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        names: list[str] = []
        oof_probas: list[np.ndarray] = []
        oof_f1: list[float] = []
        for name, est in self.estimators:
            proba = cross_val_predict(clone(est), X, y_arr, cv=cv, method="predict_proba", n_jobs=-1)
            proba = np.asarray(proba, dtype=np.float64)
            pred = self.classes_[np.argmax(proba, axis=1)]
            f1_val = float(f1_score(y_arr, pred, average="macro"))
            names.append(name)
            oof_probas.append(proba)
            oof_f1.append(f1_val)

        f1_arr = np.asarray(oof_f1, dtype=np.float64)
        max_f1 = float(f1_arr.max()) if f1_arr.size else 0.0
        gaps = max_f1 - f1_arr
        prune_delta = float(eff["prune_delta"])
        keep_mask = gaps <= prune_delta
        if int(keep_mask.sum()) < int(self.min_keep):
            top_idx = np.argsort(f1_arr)[-int(self.min_keep) :]
            keep_mask = np.zeros_like(keep_mask, dtype=bool)
            keep_mask[top_idx] = True
        keep_idx = np.where(keep_mask)[0]
        self.kept_base_names_ = [names[i] for i in keep_idx]
        self.pruned_base_names_ = [names[i] for i in range(len(names)) if i not in keep_idx]

        kept_f1 = f1_arr[keep_idx]
        temperature = float(eff["temperature"])
        scaled = (kept_f1 - kept_f1.max()) / max(temperature, 1e-6)
        global_w = np.exp(scaled)
        global_w = global_w / np.clip(global_w.sum(), 1e-12, None)
        self.global_skill_weights_ = {
            names[i]: float(w) for i, w in zip(keep_idx, global_w, strict=False)
        }
        self.base_macro_f1_ = {n: float(f) for n, f in zip(names, oof_f1, strict=False)}

        kept_oof_probas = [oof_probas[i] for i in keep_idx]
        margins = np.column_stack([_margin_from_proba(p) for p in kept_oof_probas])
        sample_w = self._per_sample_weights(margins, global_w)
        p_oof_weighted = np.zeros_like(kept_oof_probas[0], dtype=np.float64)
        for idx, p in enumerate(kept_oof_probas):
            p_oof_weighted += sample_w[:, idx : idx + 1] * p

        # Stacked fusion: OOF meta-CV so selection does not leak test data.
        fusion_mode = str(eff.get("fusion", self.fusion) or "weighted").lower()
        stack_features_oof = self._stack_features(kept_oof_probas)
        p_oof_stacked: np.ndarray | None = None
        meta_estimator = None
        if fusion_mode in {"stacked", "auto"}:
            meta_inner_splits = max(2, min(int(self.stack_meta_cv), n_min))
            meta_cv = StratifiedKFold(
                n_splits=meta_inner_splits, shuffle=True, random_state=self.random_state
            )
            meta_template = LogisticRegression(
                C=float(self.stack_meta_C),
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state,
            )
            try:
                p_oof_stacked = np.asarray(
                    cross_val_predict(
                        clone(meta_template),
                        stack_features_oof,
                        y_arr,
                        cv=meta_cv,
                        method="predict_proba",
                        n_jobs=-1,
                    ),
                    dtype=np.float64,
                )
                meta_estimator = clone(meta_template)
                meta_estimator.fit(stack_features_oof, y_arr)
            except (ValueError, RuntimeError):
                p_oof_stacked = None
                meta_estimator = None

        weighted_f1 = float(
            f1_score(
                y_arr,
                self.classes_[np.argmax(p_oof_weighted, axis=1)],
                average="macro",
            )
        )
        stacked_f1 = -float("inf")
        if p_oof_stacked is not None:
            stacked_f1 = float(
                f1_score(
                    y_arr,
                    self.classes_[np.argmax(p_oof_stacked, axis=1)],
                    average="macro",
                )
            )

        if fusion_mode == "stacked" and p_oof_stacked is not None:
            chosen_fusion = "stacked"
        elif fusion_mode == "auto" and p_oof_stacked is not None and (
            stacked_f1 > weighted_f1 + float(self.stack_select_margin)
        ):
            chosen_fusion = "stacked"
        else:
            chosen_fusion = "weighted"
        self.fusion_used_ = chosen_fusion
        self.fusion_oof_f1_ = {"weighted": weighted_f1, "stacked": stacked_f1}

        if chosen_fusion == "stacked" and p_oof_stacked is not None:
            p_oof = p_oof_stacked
            self.meta_estimator_ = meta_estimator
        else:
            p_oof = p_oof_weighted
            self.meta_estimator_ = None

        # Recalibrate fused OOF probs (binary); OOF-only so no test leakage.
        self.fusion_recalibrator_ = None
        self.fusion_recalibration_method_ = None
        if (
            self.recalibrate_fusion
            and len(self.classes_) == 2
            and p_oof.shape[1] == 2
        ):
            method = self.recalibration_method
            if method == "auto":
                method = "isotonic" if n_train >= 1500 else "sigmoid"
            try:
                if method == "isotonic":
                    from sklearn.isotonic import IsotonicRegression

                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(p_oof[:, 1], y_arr.astype(float))
                    self.fusion_recalibrator_ = iso
                else:
                    from sklearn.linear_model import LogisticRegression as _LR

                    platt = _LR(C=1.0, solver="lbfgs", max_iter=2000)
                    platt.fit(p_oof[:, 1].reshape(-1, 1), y_arr.astype(int))
                    self.fusion_recalibrator_ = platt
                self.fusion_recalibration_method_ = method
                p_oof = self._apply_fusion_recalibration(p_oof)
            except (ValueError, RuntimeError):
                self.fusion_recalibrator_ = None
                self.fusion_recalibration_method_ = None

        self.oof_ensemble_proba_ = p_oof

        self.threshold_ = 0.5
        if self.tune_threshold and len(self.classes_) == 2:
            grid = list(eff["threshold_grid"])
            f1_weight = float(eff["f1_weight"])
            mcc_weight = float(eff["mcc_weight"])
            obj = dict(self.objective_weights or {})
            dp_weight = float(obj.get("dp_diff", 0.0))
            eod_weight = float(obj.get("eod_diff", 0.0))
            fair_penalty_scale = float(self.fairness_threshold_penalty)
            use_fairness = (
                fair_penalty_scale > 0.0
                and sensitive_arr is not None
                and (dp_weight > 0.0 or eod_weight > 0.0)
            )
            best_score = -float("inf")
            for thr in grid:
                y_pred = (p_oof[:, 1] >= float(thr)).astype(int)
                score = _joint_perf_score_binary(
                    y_arr,
                    y_pred,
                    f1_weight=f1_weight,
                    mcc_weight=mcc_weight,
                )
                if use_fairness:
                    dp_gap, eod_gap = _fairness_gap_binary(
                        y_arr, y_pred, sensitive_arr
                    )
                    score -= fair_penalty_scale * (
                        dp_weight * dp_gap + eod_weight * eod_gap
                    )
                if score > best_score:
                    best_score = score
                    self.threshold_ = float(thr)
            lo, hi = self.threshold_clip
            self.threshold_ = float(np.clip(self.threshold_, float(lo), float(hi)))

        fitted: list[tuple[str, object]] = []
        for i in keep_idx:
            est = clone(self.estimators[i][1])
            if sample_weight is None:
                est.fit(X, y_arr)
            else:
                est.fit(X, y_arr, sample_weight=sample_weight)
            fitted.append((names[i], est))
        self.fitted_estimators_ = fitted
        return self

    def _apply_fusion_recalibration(self, proba: np.ndarray) -> np.ndarray:
        """Recalibrate binary fused probabilities using the fitted recalibrator."""
        recal = getattr(self, "fusion_recalibrator_", None)
        if recal is None or proba.ndim != 2 or proba.shape[1] != 2:
            return proba
        pos = np.clip(np.asarray(proba[:, 1], dtype=np.float64), 0.0, 1.0)
        if hasattr(recal, "predict") and not hasattr(recal, "predict_proba"):
            cal_pos = np.clip(np.asarray(recal.predict(pos), dtype=np.float64), 0.0, 1.0)
        else:
            cal_pos = np.clip(
                np.asarray(recal.predict_proba(pos.reshape(-1, 1))[:, 1], dtype=np.float64),
                0.0,
                1.0,
            )
        out = np.zeros_like(proba, dtype=np.float64)
        out[:, 1] = cal_pos
        out[:, 0] = 1.0 - cal_pos
        return out

    @staticmethod
    def _stack_features(kept_oof_probas: list[np.ndarray]) -> np.ndarray:
        # Drop last binary column (linearly dependent); keep all columns for k>2.
        cols = []
        for p in kept_oof_probas:
            arr = np.asarray(p, dtype=np.float64)
            if arr.ndim != 2:
                arr = arr.reshape(-1, 1)
            if arr.shape[1] == 2:
                cols.append(arr[:, 1:2])
            else:
                cols.append(arr)
        return np.concatenate(cols, axis=1)

    def predict_proba(self, X):
        probas = [np.asarray(est.predict_proba(X), dtype=np.float64) for _, est in self.fitted_estimators_]
        chosen = getattr(self, "fusion_used_", "weighted")
        meta = getattr(self, "meta_estimator_", None)
        if chosen == "stacked" and meta is not None:
            features = self._stack_features(probas)
            out = np.asarray(meta.predict_proba(features), dtype=np.float64)
        else:
            margins = np.column_stack([_margin_from_proba(p) for p in probas])
            global_w = np.asarray(
                [self.global_skill_weights_[name] for name, _ in self.fitted_estimators_],
                dtype=np.float64,
            )
            sample_w = self._per_sample_weights(margins, global_w)
            out = np.zeros_like(probas[0], dtype=np.float64)
            for idx, p in enumerate(probas):
                out += sample_w[:, idx : idx + 1] * p
        return self._apply_fusion_recalibration(out)

    def predict(self, X):
        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            return (proba[:, 1] >= self.threshold_).astype(int)
        return self.classes_[np.argmax(proba, axis=1)]


def _build_hies(seed: int, class_weight_ratio: float, mp: dict | None) -> _HIESAdaptive:
    """HIES: OOF-pruned, per-sample confidence-weighted calibrated ensemble."""
    cfg = dict(mp or {})
    prune_delta = float(cfg.pop("prune_delta", 0.025))
    min_keep = int(cfg.pop("min_keep", 3))
    temperature = float(cfg.pop("temperature", 0.08))
    margin_floor = float(cfg.pop("margin_floor", 0.05))
    cv_splits = int(cfg.pop("cv_splits", 5))
    calibration_cv = int(cfg.pop("calibration_cv", 3))
    calibration_method = str(cfg.pop("calibration_method", "auto"))
    n_train_threshold = int(cfg.pop("n_train_threshold", 1500))
    tune_threshold = bool(cfg.pop("tune_threshold", True))
    threshold_grid = cfg.pop("threshold_grid", [0.40, 0.43, 0.45, 0.47, 0.50, 0.53, 0.55])
    threshold_clip = cfg.pop("threshold_clip", [0.30, 0.70])
    objective_weights = cfg.pop("objective_weights", {"f1_macro": 0.88, "mcc": 0.50, "dp_diff": 0.03, "eod_diff": 0.02})
    fusion = str(cfg.pop("fusion", "weighted"))
    stack_meta_C = float(cfg.pop("stack_meta_C", 1.0))
    stack_meta_cv = int(cfg.pop("stack_meta_cv", 5))
    stack_select_margin = float(cfg.pop("stack_select_margin", 0.0))
    auto_cfg = dict(cfg.pop("auto_config", {}))
    auto_enabled = bool(auto_cfg.pop("enabled", False))
    auto_n_large = int(auto_cfg.pop("n_large_split", 2000))
    auto_imb_thresh = float(auto_cfg.pop("imbalance_split", 2.5))
    auto_small = auto_cfg.pop("small", None)
    auto_large = auto_cfg.pop("large", None)
    raw_dim_split = auto_cfg.pop("feature_dim_split", None)
    auto_feature_dim_split = int(raw_dim_split) if raw_dim_split is not None else None
    auto_wide = auto_cfg.pop("wide", None)
    auto_narrow = auto_cfg.pop("narrow", None)
    auto_imb_boost = float(auto_cfg.pop("imbalanced_mcc_boost", 0.0))
    if auto_cfg:
        raise ValueError(f"Unknown keys under auto_config: {sorted(auto_cfg)}")
    xgb_kw = dict(cfg.pop("xgb", {}))
    rf_kw = dict(cfg.pop("rf", {}))
    hgb_kw = dict(cfg.pop("hgb", {}))
    lgbm_kw = dict(cfg.pop("lgbm", {}))
    cat_kw = dict(cfg.pop("cat", {}))
    include_catboost = bool(cfg.pop("include_catboost", True))
    recalibrate_fusion = bool(cfg.pop("recalibrate_fusion", True))
    recalibration_method = str(cfg.pop("recalibration_method", "auto"))
    recalibration_cv = int(cfg.pop("recalibration_cv", 5))
    fairness_threshold_penalty = float(cfg.pop("fairness_threshold_penalty", 0.0))
    if cfg:
        raise ValueError(f"Unknown keys for hies: {sorted(cfg)}")
    bases = _heterogeneous_calibrated_bases(
        seed=seed,
        class_weight_ratio=class_weight_ratio,
        xgb_params=xgb_kw,
        rf_params=rf_kw,
        hgb_params=hgb_kw,
        lgbm_params=lgbm_kw,
        cat_params=cat_kw,
        include_catboost=include_catboost,
        calibration_cv=calibration_cv,
        calibration_method=calibration_method,
        n_train_threshold=n_train_threshold,
    )
    return _HIESAdaptive(
        estimators=bases,
        prune_delta=prune_delta,
        min_keep=min_keep,
        temperature=temperature,
        margin_floor=margin_floor,
        cv_splits=cv_splits,
        tune_threshold=tune_threshold,
        threshold_grid=list(threshold_grid),
        threshold_clip=tuple(threshold_clip),
        objective_weights=objective_weights,
        auto_config=auto_enabled,
        auto_n_large=auto_n_large,
        auto_imb_thresh=auto_imb_thresh,
        auto_small=auto_small,
        auto_large=auto_large,
        auto_feature_dim_split=auto_feature_dim_split,
        auto_wide=auto_wide,
        auto_narrow=auto_narrow,
        auto_imbalanced_mcc_boost=auto_imb_boost,
        fusion=fusion,
        stack_meta_C=stack_meta_C,
        stack_meta_cv=stack_meta_cv,
        stack_select_margin=stack_select_margin,
        recalibrate_fusion=recalibrate_fusion,
        recalibration_method=recalibration_method,
        recalibration_cv=recalibration_cv,
        fairness_threshold_penalty=fairness_threshold_penalty,
        random_state=seed,
    )


def make_preprocessor(X):
    numeric_cols = [c for c in X.columns if X[c].dtype.kind in "if"]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def compute_fair_reweighting(y, sensitive, min_weight: float = 0.5, max_weight: float = 3.0):
    """Return sample weights balancing label-sensitive intersections.

    This approximates fairness reweighing using inverse-frequency weighting
    over (label, sensitive group) cells while keeping weights numerically stable.
    """

    y_series = pd.Series(np.asarray(y)).reset_index(drop=True)
    s_series = pd.Series(np.asarray(sensitive)).reset_index(drop=True).astype(str)
    frame = pd.DataFrame({"_y": y_series, "_s": s_series})
    cell_freq = frame.value_counts(normalize=True)
    if cell_freq.empty:
        return np.ones(len(frame), dtype=float)
    raw_weights = []
    for y_val, s_val in zip(frame["_y"], frame["_s"]):
        freq = float(cell_freq.get((y_val, s_val), 0.0))
        weight = 1.0 / max(freq, 1e-12)
        raw_weights.append(weight)
    weights = np.asarray(raw_weights, dtype=float)
    weights = weights / float(np.mean(weights))
    return np.clip(weights, min_weight, max_weight)


def find_group_thresholds(
    y_true,
    y_prob,
    sensitive,
    threshold_grid: list[float],
    score_fn,
    baseline_dp: float | None = None,
    baseline_eod: float | None = None,
    guardrail_fn=None,
    max_dp_increase: float = 0.03,
    max_eod_increase: float = 0.03,
):
    """Find group-specific thresholds by exhaustive search on a small grid."""

    groups = sorted(pd.Series(np.asarray(sensitive).astype(str)).unique().tolist())
    if len(groups) < 2:
        return {"__default__": 0.5}

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    sensitive = np.asarray(sensitive).astype(str)
    best_map = {group: 0.5 for group in groups}
    best_score = -float("inf")
    for combo in product(threshold_grid, repeat=len(groups)):
        thr_map = {group: float(thr) for group, thr in zip(groups, combo)}
        y_pred = np.zeros_like(y_true, dtype=int)
        for group in groups:
            mask = sensitive == group
            if mask.any():
                y_pred[mask] = (y_prob[mask] >= thr_map[group]).astype(int)
        score = float(score_fn(y_true, y_pred, y_prob, sensitive))
        if guardrail_fn is not None:
            fair = guardrail_fn(y_true, y_pred, sensitive)
            if baseline_dp is not None and baseline_eod is not None:
                if not (
                    fair["demographic_parity_diff"] <= baseline_dp + max_dp_increase
                    and fair["equal_opportunity_diff"] <= baseline_eod + max_eod_increase
                ):
                    score -= 1.0
        if score > best_score:
            best_score = score
            best_map = thr_map
    return best_map


def build_method(
    name: str,
    seed: int,
    class_weight_ratio: float,
    method_params: dict | None = None,
) -> MethodSpec:
    """Factory for benchmark methods (IDs align with manuscript naming)."""

    params = method_params or {}
    xgb_params = params.get(name, {})

    if name == "logistic_regression":
        lr_params = params.get(name, {})
        return MethodSpec(name, _base_lr(seed, model_params=lr_params), None)

    if name == "xgboost":
        return MethodSpec(name, _base_xgb(seed, model_params=xgb_params), None)

    if name == "calibrated_xgboost":
        return MethodSpec(
            name,
            _calibrated_xgb(seed, class_weight_ratio=class_weight_ratio, model_params=xgb_params),
            None,
        )

    if name == "lightgbm":
        lgb_params = params.get(name, {})
        return MethodSpec(
            name,
            _base_lgbm(seed, class_weight_ratio=class_weight_ratio, model_params=lgb_params),
            None,
        )

    if name == "fairlearn_equalized_odds_xgb":
        return MethodSpec(
            name,
            _PredictProbaFloat64Wrapper(_base_xgb(seed, model_params=xgb_params)),
            None,
            False,
            False,
            True,
        )

    if name == "hies":
        mp = params.get(name, {})
        return MethodSpec(name, _build_hies(seed, class_weight_ratio, mp), None)

    if name == "catboost":
        cb_params = params.get(name, {})
        return MethodSpec(
            name,
            _base_catboost(seed, class_weight_ratio=class_weight_ratio, model_params=cb_params),
            None,
        )

    if name == "extra_trees":
        et_params = params.get(name, {})
        merged = {
            "n_estimators": 600,
            "max_depth": None,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": seed,
        }
        merged.update(et_params)
        return MethodSpec(name, ExtraTreesClassifier(**merged), None)

    if name == "tabpfn":
        tp_params = params.get(name, {})
        max_n_train = int(tp_params.get("max_n_train", 2000))
        return MethodSpec(name, _TabPFNAdapter(seed=seed, max_n_train=max_n_train), None)

    if name == "random_forest":
        rf_params = params.get(name, {})
        merged = {
            "n_estimators": 600,
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": seed,
        }
        merged.update(rf_params)
        return MethodSpec(name, RandomForestClassifier(**merged), None)

    if name == "voting":
        vt_params = params.get(name, {})
        bases_raw = [
            ("xgb", _base_xgb(seed, class_weight=class_weight_ratio)),
            ("lgbm", _base_lgbm(seed, class_weight_ratio=class_weight_ratio)),
            ("rf", RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )),
            ("hgb", HistGradientBoostingClassifier(
                max_iter=260,
                max_depth=6,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=seed,
            )),
        ]
        if CatBoostClassifier is not None:
            bases_raw.append((
                "cat",
                CatBoostClassifier(
                    iterations=int(vt_params.get("cat_iterations", 280)),
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=3.0,
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                    auto_class_weights="Balanced",
                ),
            ))
        return MethodSpec(
            name,
            VotingClassifier(
                estimators=bases_raw,
                voting="soft",
                n_jobs=None,  # CatBoost is not safely picklable with joblib.
            ),
            None,
        )

    if name == "stacking":
        st_params = params.get(name, {})
        bases_raw = [
            ("xgb", _base_xgb(seed, class_weight=class_weight_ratio)),
            ("lgbm", _base_lgbm(seed, class_weight_ratio=class_weight_ratio)),
            ("rf", RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )),
            ("hgb", HistGradientBoostingClassifier(
                max_iter=260,
                max_depth=6,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=seed,
            )),
        ]
        if CatBoostClassifier is not None:
            bases_raw.append((
                "cat",
                CatBoostClassifier(
                    iterations=int(st_params.get("cat_iterations", 280)),
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=3.0,
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                    auto_class_weights="Balanced",
                ),
            ))
        meta = LogisticRegression(
            C=float(st_params.get("meta_C", 1.0)),
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
        return MethodSpec(
            name,
            StackingClassifier(
                estimators=bases_raw,
                final_estimator=meta,
                cv=int(st_params.get("cv", 5)),
                stack_method="predict_proba",
                passthrough=False,
                n_jobs=None,
            ),
            None,
        )

    if name == "stacking_rf_catboost":
        if CatBoostClassifier is None:
            raise ImportError("stacking_rf_catboost requires catboost to be installed")
        st_params = params.get(name, {})
        bases_raw = [
            ("rf", RandomForestClassifier(
                n_estimators=int(st_params.get("rf_n_estimators", 600)),
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )),
            ("cat", CatBoostClassifier(
                iterations=int(st_params.get("cat_iterations", 400)),
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
                auto_class_weights="Balanced",
            )),
        ]
        meta = LogisticRegression(
            C=float(st_params.get("meta_C", 1.0)),
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
        return MethodSpec(
            name,
            StackingClassifier(
                estimators=bases_raw,
                final_estimator=meta,
                cv=int(st_params.get("cv", 5)),
                stack_method="predict_proba",
                passthrough=False,
                n_jobs=None,
            ),
            None,
        )

    raise ValueError(f"Unknown method: {name}")
