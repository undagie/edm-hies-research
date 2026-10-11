import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.models.methods import _HIESAdaptive, build_method


@pytest.fixture
def xy_small():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6))
    y = (rng.random(80) > 0.65).astype(int)
    return X, y


@pytest.fixture
def xy_medium():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    y = (rng.random(200) > 0.6).astype(int)
    return X, y


@pytest.mark.parametrize("name", ["random_forest", "voting", "stacking"])
def test_classical_ensemble_baselines_fit_and_predict_proba(name, xy_medium):
    """RF/Voting/Stacking predict_proba returns shape (n, 2) rows summing to 1."""
    X, y = xy_medium
    spec = build_method(
        name,
        seed=1,
        class_weight_ratio=2.0,
        method_params={
            "voting": {"cat_iterations": 60},
            "stacking": {"cat_iterations": 60, "cv": 3, "meta_C": 1.0},
        },
    )
    spec.estimator.fit(X, y)
    proba = spec.estimator.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)


def test_unknown_param_keys_raise_hies():
    with pytest.raises(ValueError, match="Unknown keys"):
        build_method(
            "hies",
            seed=1,
            class_weight_ratio=2.0,
            method_params={"hies": {"bogus": 1}},
        )


def test_hies_resolve_config_feature_dim_routing():
    """Wide/narrow prune_delta overlays apply by post-encoding feature dim."""
    est = [
        ("a", DummyClassifier()),
        ("b", DummyClassifier()),
        ("c", DummyClassifier()),
        ("d", DummyClassifier()),
    ]
    grid = [0.5]
    clf = _HIESAdaptive(
        est,
        auto_config=True,
        auto_n_large=100_000,
        auto_small={
            "prune_delta": 0.5,
            "temperature": 0.1,
            "mcc_weight": 0.5,
            "threshold_grid": grid,
        },
        auto_large={
            "prune_delta": 0.5,
            "temperature": 0.1,
            "mcc_weight": 0.5,
            "threshold_grid": grid,
        },
        auto_feature_dim_split=50,
        auto_wide={"prune_delta": 0.07},
        auto_narrow={"prune_delta": 0.02},
        auto_imb_thresh=100.0,
    )
    assert clf._resolve_effective_config(50, 2.0, n_features=80)["prune_delta"] == 0.07
    assert clf._resolve_effective_config(50, 2.0, n_features=10)["prune_delta"] == 0.02

    off = _HIESAdaptive(
        est,
        auto_config=True,
        auto_n_large=100_000,
        auto_small={
            "prune_delta": 0.33,
            "temperature": 0.1,
            "mcc_weight": 0.5,
            "threshold_grid": grid,
        },
        auto_large={
            "prune_delta": 0.33,
            "temperature": 0.1,
            "mcc_weight": 0.5,
            "threshold_grid": grid,
        },
        auto_feature_dim_split=None,
        auto_wide={"prune_delta": 0.07},
        auto_narrow={"prune_delta": 0.02},
        auto_imb_thresh=100.0,
    )
    assert off._resolve_effective_config(50, 2.0, n_features=999)["prune_delta"] == 0.33
