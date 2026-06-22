import pickle

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

from warpgbm import WarpGBM


def _tree_features(tree):
    features = []
    if isinstance(tree, list):
        for class_tree in tree:
            features.extend(_tree_features(class_tree))
        return features
    if not tree or "leaf_value" in tree:
        return features
    features.append(tree["feature"])
    features.extend(_tree_features(tree["left"]))
    features.extend(_tree_features(tree["right"]))
    return features


def test_feature_blinder_constructor_defaults():
    model = WarpGBM()

    assert model.feature_blinder is False
    assert model.blind_ratio == 0.0
    assert model.blind_frequency == 1.0
    assert model.blind_seed is None


@pytest.mark.parametrize(
    "kwargs, expected_error",
    [
        ({"feature_blinder": 1}, TypeError),
        ({"feature_blinder": "yes"}, TypeError),
        ({"blind_ratio": -0.01}, ValueError),
        ({"blind_ratio": 1.0}, ValueError),
        ({"blind_ratio": "0.5"}, TypeError),
        ({"blind_ratio": False}, TypeError),
        ({"blind_ratio": True}, TypeError),
        ({"blind_frequency": -0.01}, ValueError),
        ({"blind_frequency": 1.01}, ValueError),
        ({"blind_frequency": "1.0"}, TypeError),
        ({"blind_frequency": False}, TypeError),
        ({"blind_frequency": True}, TypeError),
        ({"blind_seed": 1.5}, TypeError),
        ({"blind_seed": True}, TypeError),
    ],
)
def test_feature_blinder_constructor_validation(kwargs, expected_error):
    with pytest.raises(expected_error):
        WarpGBM(**kwargs)


def _configured_helper_model(**kwargs):
    model = WarpGBM(device="cuda", **kwargs)
    model.num_features = 4
    model.num_eras = 1
    model.feature_indices = torch.arange(4, device=model.device, dtype=torch.int32)
    model.per_era_feature_importance_gpu = torch.zeros(
        (1, 4), device=model.device, dtype=torch.float32
    )
    return model


def test_feature_blinder_candidate_selection_keeps_one_feature():
    model = _configured_helper_model(
        feature_blinder=True,
        blind_ratio=0.99,
        blind_frequency=1.0,
        blind_seed=7,
    )

    selected = model._select_tree_features(round_idx=0)

    assert selected.numel() == 1
    assert selected.dtype == torch.int32
    assert 0 <= int(selected[0].item()) < 4


def test_feature_blinder_prefers_blinding_important_feature():
    model = _configured_helper_model(
        feature_blinder=True,
        blind_ratio=0.25,
        blind_frequency=1.0,
        blind_seed=7,
    )
    model.per_era_feature_importance_gpu[0, 1] = 100.0

    selected = model._select_tree_features(round_idx=0).detach().cpu().numpy()

    assert 1 not in selected
    assert set(selected).issubset({0, 2, 3})
    assert len(selected) == 3


def test_feature_blinder_frequency_zero_preserves_candidates():
    model = _configured_helper_model(
        feature_blinder=True,
        blind_ratio=0.75,
        blind_frequency=0.0,
        blind_seed=7,
    )

    selected = model._select_tree_features(round_idx=0)

    np.testing.assert_array_equal(
        selected.detach().cpu().numpy(),
        np.array([0, 1, 2, 3], dtype=np.int32),
    )


def test_feature_blinder_round_seed_is_reproducible():
    model_a = _configured_helper_model(
        feature_blinder=True,
        blind_ratio=0.5,
        blind_frequency=1.0,
        blind_seed=19,
    )
    model_b = _configured_helper_model(
        feature_blinder=True,
        blind_ratio=0.5,
        blind_frequency=1.0,
        blind_seed=19,
    )

    selected_a = model_a._select_tree_features(round_idx=3)
    selected_b = model_b._select_tree_features(round_idx=3)

    np.testing.assert_array_equal(
        selected_a.detach().cpu().numpy(),
        selected_b.detach().cpu().numpy(),
    )


def test_feature_blinder_regression_fit_uses_blinded_candidates():
    X, y = make_regression(
        n_samples=240,
        n_features=8,
        n_informative=4,
        noise=0.1,
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model = WarpGBM(
        objective="regression",
        n_estimators=8,
        max_depth=2,
        num_bins=16,
        min_child_weight=2,
        feature_blinder=True,
        blind_ratio=0.75,
        blind_frequency=1.0,
        blind_seed=42,
        random_state=42,
    )
    model.fit(X, y)

    used_features = set()
    for tree in model.forest:
        used_features.update(_tree_features(tree))

    assert model._trees_trained == 8
    assert used_features
    assert all(0 <= feature < X.shape[1] for feature in used_features)


def test_feature_blinder_disabled_matches_existing_regression_behavior():
    X, y = make_regression(
        n_samples=180,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=11,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    base = WarpGBM(
        objective="regression",
        n_estimators=6,
        max_depth=2,
        num_bins=16,
        min_child_weight=2,
        colsample_bytree=0.5,
        random_state=11,
    )
    disabled = WarpGBM(
        objective="regression",
        n_estimators=6,
        max_depth=2,
        num_bins=16,
        min_child_weight=2,
        colsample_bytree=0.5,
        random_state=11,
        feature_blinder=True,
        blind_ratio=0.0,
    )

    base.fit(X, y)
    disabled.fit(X, y)

    np.testing.assert_allclose(
        base.predict(X),
        disabled.predict(X),
        rtol=1e-3,
        atol=1e-3,
    )
