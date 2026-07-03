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
        ({"blind_ratio": float("nan")}, ValueError),
        ({"blind_ratio": float("inf")}, ValueError),
        ({"blind_ratio": float("-inf")}, ValueError),
        ({"blind_ratio": "0.5"}, TypeError),
        ({"blind_ratio": False}, TypeError),
        ({"blind_ratio": True}, TypeError),
        ({"blind_frequency": -0.01}, ValueError),
        ({"blind_frequency": 1.01}, ValueError),
        ({"blind_frequency": float("nan")}, ValueError),
        ({"blind_frequency": float("inf")}, ValueError),
        ({"blind_frequency": float("-inf")}, ValueError),
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


def test_feature_blinder_accepts_numpy_integer_seed():
    model = WarpGBM(blind_seed=np.int64(42))

    assert model.blind_seed == np.int64(42)


def test_feature_blinder_round_seed_does_not_overflow():
    model = WarpGBM(
        device="cpu",
        feature_blinder=True,
        blind_ratio=0.25,
        blind_frequency=1.0,
        blind_seed=2**64 - 1,
    )
    model.num_features = 4
    model.feature_indices = torch.arange(4, device=model.device, dtype=torch.int32)

    generator = model._feature_selection_generator(round_idx=1)

    assert generator is not None


def test_feature_blinder_does_not_drive_colsample_rng():
    def selected(random_state, blind_seed):
        torch.manual_seed(random_state)
        model = WarpGBM(
            device="cpu",
            colsample_bytree=0.5,
            random_state=random_state,
            feature_blinder=True,
            blind_ratio=0.01,
            blind_frequency=1.0,
            blind_seed=blind_seed,
        )
        model.num_features = 10
        model.feature_indices = torch.arange(10, device=model.device, dtype=torch.int32)
        return model._select_tree_features(round_idx=0).detach().cpu().numpy()

    np.testing.assert_array_equal(selected(123, 7), selected(123, 99))

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(selected(123, 7), selected(999, 7))


@pytest.mark.parametrize(
    "num_eras, buffer_name",
    [
        (1, "per_era_gain"),
        (2, "fused_gain"),
    ],
)
def test_prepare_split_buffers_reuses_matching_shape_and_updates_view_on_change(
    num_eras, buffer_name
):
    model = WarpGBM(device="cpu", num_bins=10)
    model.num_eras = num_eras

    model._prepare_split_buffers(candidate_count=4)
    first_buffer = getattr(model, buffer_name)

    model._prepare_split_buffers(candidate_count=4)
    matching_buffer = getattr(model, buffer_name)

    model._prepare_split_buffers(candidate_count=3)
    resized_buffer = getattr(model, buffer_name)

    assert matching_buffer is first_buffer
    assert resized_buffer is not first_buffer


def test_prepare_split_buffers_reuses_larger_capacity_without_reallocation():
    model = WarpGBM(device="cpu", num_bins=10)
    model.num_eras = 1

    model._prepare_split_buffers(candidate_count=4)
    first_data_ptr = model.per_era_gain.data_ptr()

    model._prepare_split_buffers(candidate_count=3)

    assert model.per_era_gain.shape == (1, 3, 9)
    assert model.per_era_gain.data_ptr() == first_data_ptr


def test_prepare_split_buffers_does_not_zero_matching_shape():
    model = WarpGBM(device="cpu", num_bins=10)
    model.num_eras = 1

    model._prepare_split_buffers(candidate_count=4)
    model.per_era_gain.fill_(123.0)
    model.per_era_direction.fill_(456.0)

    model._prepare_split_buffers(candidate_count=4)

    assert torch.all(model.per_era_gain == 123.0)
    assert torch.all(model.per_era_direction == 456.0)


def test_identity_feature_mapping_does_not_read_feature_tensor():
    class ExplodingFeatureTensor:
        def __getitem__(self, _):
            raise AssertionError("identity mapping should not index feat_indices_tree")

    model = WarpGBM(device="cpu")
    model._feat_indices_are_identity = True
    model.feat_indices_tree = ExplodingFeatureTensor()

    assert model._global_feature_for_local(3) == 3


def test_root_hessian_reuse_survives_skipped_blinding_events():
    model = WarpGBM(
        device="cpu",
        n_estimators=4,
        num_bins=6,
        colsample_bytree=1.0,
        feature_blinder=True,
        blind_ratio=0.5,
        blind_frequency=0.5,
        blind_seed=7,
    )
    model.num_features = 6
    model.num_eras = 1
    model.feature_indices = torch.arange(6, device=model.device, dtype=torch.int32)
    model.root_node_indices = torch.arange(4, device=model.device, dtype=torch.int32)
    model.Y_gpu = torch.ones(4, device=model.device, dtype=torch.float32)
    model.gradients = torch.zeros(4, device=model.device, dtype=torch.float32)
    model.era_sorted_for_hist = True
    model.max_root_era_count = 4
    model.forest = [{} for _ in range(model.n_estimators)]

    full_histogram_calls = 0
    root_grad_only_calls = 0

    def fake_compute_histograms(_, feature_indices):
        nonlocal full_histogram_calls
        full_histogram_calls += 1
        feature_count = int(feature_indices.numel())
        return (
            torch.zeros(
                (1, feature_count, model.num_bins),
                device=model.device,
                dtype=torch.float32,
            ),
            torch.ones(
                (1, feature_count, model.num_bins),
                device=model.device,
                dtype=torch.float32,
            ),
        )

    def fake_root_gradient_histogram_by_era():
        nonlocal root_grad_only_calls
        root_grad_only_calls += 1
        return torch.zeros(
            (1, model.num_features, model.num_bins),
            device=model.device,
            dtype=torch.float32,
        )

    model.compute_histograms = fake_compute_histograms
    model.compute_root_gradient_histogram_by_era = fake_root_gradient_histogram_by_era
    model.grow_tree = lambda *args, **kwargs: {"leaf_value": torch.tensor(0.0), "samples": 4}
    model.compute_eval = lambda _: None

    model.grow_forest()

    assert full_histogram_calls == 2
    assert root_grad_only_calls == 2


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
        num_bins=10,
        min_child_weight=2,
        feature_blinder=True,
        blind_ratio=0.75,
        blind_frequency=1.0,
        blind_seed=42,
        random_state=42,
    )
    selected_rounds = []
    original_select_tree_features = model._select_tree_features

    def select_tree_features_spy(round_idx):
        selected_rounds.append(round_idx)
        return original_select_tree_features(round_idx)

    model._select_tree_features = select_tree_features_spy

    model.fit(X, y)

    used_features = set()
    for tree in model.forest:
        used_features.update(_tree_features(tree))

    assert model._trees_trained == 8
    assert selected_rounds == list(range(model._trees_trained))
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
        num_bins=10,
        min_child_weight=2,
        colsample_bytree=0.5,
        random_state=11,
    )
    disabled = WarpGBM(
        objective="regression",
        n_estimators=6,
        max_depth=2,
        num_bins=10,
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


def test_feature_blinder_multiclass_fit_runs():
    X, y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
    X = X.astype(np.float32)

    model = WarpGBM(
        objective="multiclass",
        n_estimators=5,
        max_depth=2,
        num_bins=10,
        min_child_weight=2,
        feature_blinder=True,
        blind_ratio=0.5,
        blind_frequency=1.0,
        blind_seed=42,
        random_state=42,
    )
    model.fit(X, y)
    probabilities = model.predict_proba(X)

    assert model._trees_trained == 5
    assert probabilities.shape == (X.shape[0], 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_feature_blinder_multiclass_selects_candidates_per_class_tree():
    model = WarpGBM(
        objective="multiclass",
        device="cpu",
        n_estimators=1,
        num_bins=6,
        colsample_bytree=1.0,
        feature_blinder=True,
        blind_ratio=0.5,
        blind_frequency=1.0,
        blind_seed=7,
    )
    model.num_classes = 3
    model.num_features = 6
    model.num_eras = 1
    model.feature_indices = torch.arange(6, device=model.device, dtype=torch.int32)
    model.per_era_feature_importance_gpu = torch.zeros(
        (1, 6), device=model.device, dtype=torch.float32
    )
    model.Y_gpu = torch.tensor([0, 1], device=model.device, dtype=torch.int32)
    model.gradients = torch.zeros((2, 3), device=model.device, dtype=torch.float32)
    model.root_node_indices = torch.arange(2, device=model.device, dtype=torch.int32)
    model.eval_every_n_trees = None

    selected_by_tree = []

    def fake_softmax_gradients_hessians(_):
        return (
            torch.zeros((2, 3), device=model.device, dtype=torch.float32),
            torch.ones((2, 3), device=model.device, dtype=torch.float32),
        )

    def fake_histograms(_, feature_indices, grad, hess):
        feature_count = int(feature_indices.numel())
        return (
            torch.zeros(
                (1, feature_count, model.num_bins),
                device=model.device,
                dtype=torch.float32,
            ),
            torch.ones(
                (1, feature_count, model.num_bins),
                device=model.device,
                dtype=torch.float32,
            ),
        )

    def fake_grow_tree(*args, **kwargs):
        selected_by_tree.append(tuple(model.feat_indices_tree.detach().cpu().tolist()))
        return {"leaf_value": torch.tensor(0.0), "samples": 2}

    model._compute_softmax_gradients_hessians = fake_softmax_gradients_hessians
    model.compute_histograms_multiclass = fake_histograms
    model.grow_tree = fake_grow_tree
    model.compute_eval_multiclass = lambda _: None

    model.grow_forest_multiclass()

    assert len(selected_by_tree) == 3
    assert len(set(selected_by_tree)) > 1


def test_multiclass_colsample_is_shared_per_round_when_blinder_disabled():
    model = WarpGBM(
        objective="multiclass",
        device="cpu",
        n_estimators=1,
        num_bins=6,
        colsample_bytree=0.5,
        feature_blinder=False,
    )
    model.num_classes = 3
    model.num_features = 8
    model.num_eras = 1
    model.feature_indices = torch.arange(8, device=model.device, dtype=torch.int32)
    model.per_era_feature_importance_gpu = torch.zeros(
        (1, 8), device=model.device, dtype=torch.float32
    )
    model.Y_gpu = torch.tensor([0, 1], device=model.device, dtype=torch.int32)
    model.gradients = torch.zeros((2, 3), device=model.device, dtype=torch.float32)
    model.root_node_indices = torch.arange(2, device=model.device, dtype=torch.int32)
    model.eval_every_n_trees = None

    selected_by_tree = []

    def fake_softmax_gradients_hessians(_):
        return (
            torch.zeros((2, 3), device=model.device, dtype=torch.float32),
            torch.ones((2, 3), device=model.device, dtype=torch.float32),
        )

    def fake_histograms(_, feature_indices, grad, hess):
        feature_count = int(feature_indices.numel())
        return (
            torch.zeros(
                (1, feature_count, model.num_bins),
                device=model.device,
                dtype=torch.float32,
            ),
            torch.ones(
                (1, feature_count, model.num_bins),
                device=model.device,
                dtype=torch.float32,
            ),
        )

    def fake_grow_tree(*args, **kwargs):
        selected_by_tree.append(tuple(model.feat_indices_tree.detach().cpu().tolist()))
        return {"leaf_value": torch.tensor(0.0), "samples": 2}

    model._compute_softmax_gradients_hessians = fake_softmax_gradients_hessians
    model.compute_histograms_multiclass = fake_histograms
    model.grow_tree = fake_grow_tree
    model.compute_eval_multiclass = lambda _: None

    torch.manual_seed(123)
    model.grow_forest_multiclass()

    assert len(selected_by_tree) == 3
    assert selected_by_tree[0] == selected_by_tree[1] == selected_by_tree[2]


def test_feature_blinder_warm_start_continues_training():
    X, y = make_regression(
        n_samples=180,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=21,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model = WarpGBM(
        objective="regression",
        n_estimators=3,
        max_depth=2,
        num_bins=10,
        min_child_weight=2,
        feature_blinder=True,
        blind_ratio=0.5,
        blind_frequency=1.0,
        blind_seed=21,
        random_state=21,
        warm_start=True,
    )
    model.fit(X, y)
    assert model._trees_trained == 3

    model.n_estimators = 5
    model.fit(X, y)

    assert model._trees_trained == 5
    assert len(model.forest) >= 5


def test_feature_blinder_reproducible_with_fixed_seed():
    X, y = make_regression(
        n_samples=180,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=31,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    kwargs = dict(
        objective="regression",
        n_estimators=6,
        max_depth=2,
        num_bins=10,
        min_child_weight=2,
        feature_blinder=True,
        blind_ratio=0.5,
        blind_frequency=1.0,
        blind_seed=31,
        random_state=999,
    )

    model_a = WarpGBM(**kwargs)
    model_b = WarpGBM(**kwargs)
    model_a.fit(X, y)
    model_b.fit(X, y)

    np.testing.assert_allclose(model_a.predict(X), model_b.predict(X), rtol=1e-3, atol=1e-3)


def test_feature_blinder_save_load_preserves_parameters(tmp_path):
    X, y = make_regression(
        n_samples=120,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=41,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model = WarpGBM(
        objective="regression",
        n_estimators=3,
        max_depth=2,
        num_bins=10,
        min_child_weight=2,
        feature_blinder=True,
        blind_ratio=0.4,
        blind_frequency=0.5,
        blind_seed=123,
        random_state=41,
    )
    model.fit(X, y)

    path = tmp_path / "feature_blinder.pkl"
    model.save_model(path)

    loaded = WarpGBM()
    loaded.load_model(path)

    assert loaded.feature_blinder is True
    assert loaded.blind_ratio == 0.4
    assert loaded.blind_frequency == 0.5
    assert loaded.blind_seed == 123


def test_feature_blinder_loads_legacy_state_defaults(tmp_path):
    legacy_state = {
        "objective": "regression",
        "num_bins": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 1,
        "min_child_weight": 20,
        "min_split_gain": 0.0,
        "L2_reg": 1e-6,
        "colsample_bytree": 1.0,
        "random_state": None,
        "warm_start": False,
        "forest": [{}],
        "bin_edges": None,
        "base_prediction": 0.0,
        "num_features": 1,
        "num_classes": None,
        "classes_": None,
        "label_encoder": None,
        "feature_importance_": None,
        "per_era_feature_importance_": None,
        "_is_fitted": True,
        "_trees_trained": 1,
        "training_loss": [],
        "eval_loss": [],
    }
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as handle:
        pickle.dump(legacy_state, handle)

    model = WarpGBM()
    model.load_model(path)

    assert model.feature_blinder is False
    assert model.blind_ratio == 0.0
    assert model.blind_frequency == 1.0
    assert model.blind_seed is None
