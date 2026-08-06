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


def _clean_matrix(num_samples, num_features, num_bins, device="cpu", seed=0):
    """A binned matrix in which no cell already holds the blind bin.

    That makes "cell differs from clean" an exact test for "cell was blinded".
    """
    blind_bin = (num_bins - 1) // 2
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    matrix = torch.randint(
        0,
        num_bins,
        (num_samples, num_features),
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    matrix[matrix == blind_bin] = (blind_bin + 1) % num_bins
    return matrix


def _blinder_model(
    num_samples=4096,
    num_features=8,
    num_bins=10,
    device="cpu",
    importance=None,
    blind_every_n_trees=1,
    **kwargs,
):
    """A model carrying just enough training state to drive the blinder directly."""
    model = WarpGBM(
        device=device,
        num_bins=num_bins,
        blind_every_n_trees=blind_every_n_trees,
        **kwargs,
    )
    model.num_samples = num_samples
    model.num_features = num_features
    model.num_eras = 1
    model.feature_indices = torch.arange(num_features, device=device, dtype=torch.int32)

    if importance is None:
        importance = torch.zeros(num_features, device=device, dtype=torch.float32)
    model.per_era_feature_importance_gpu = importance.reshape(1, num_features).to(
        device=device, dtype=torch.float32
    )

    model.bin_indices = _clean_matrix(num_samples, num_features, num_bins, device=device)
    model._setup_blinder_buffers()
    return model


def _blinded_fraction_per_feature(model):
    changed = model.bin_indices != model.bin_indices_clean
    return changed.to(torch.float32).mean(dim=0).cpu().numpy()


# ─────────────────────────── constructor / validation ───────────────────────────


def test_feature_blinder_constructor_defaults():
    model = WarpGBM()

    assert model.feature_blinder is False
    assert model.blind_ratio == 0.0
    assert model.blind_frequency == 1.0
    assert model.blind_every_n_trees == 50
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
        ({"blind_every_n_trees": 0}, ValueError),
        ({"blind_every_n_trees": -1}, ValueError),
        ({"blind_every_n_trees": 1.5}, TypeError),
        ({"blind_every_n_trees": "10"}, TypeError),
        ({"blind_every_n_trees": True}, TypeError),
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
        blind_seed=2**64 - 1,
    )

    assert model._blind_generator(round_idx=1) is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_blinder": False, "blind_ratio": 0.5},
        {"feature_blinder": True, "blind_ratio": 0.0},
        {"feature_blinder": True, "blind_ratio": 0.5, "blind_frequency": 0.0},
    ],
)
def test_feature_blinder_inactive_combinations(kwargs):
    assert WarpGBM(device="cpu", **kwargs)._feature_blinder_active() is False


# ─────────────────────────── per-cell perturbation ───────────────────────────


@pytest.mark.parametrize(
    "num_bins, expected", [(2, 0), (3, 1), (10, 4), (11, 5), (64, 31), (127, 63)]
)
def test_blind_value_is_the_middle_bin(num_bins, expected):
    assert WarpGBM(device="cpu", num_bins=num_bins)._blind_bin() == expected


def test_blinding_writes_only_the_blind_bin():
    model = _blinder_model(feature_blinder=True, blind_ratio=0.3, blind_seed=11)

    assert model._maybe_blind_cells(round_idx=1) is True

    changed = model.bin_indices != model.bin_indices_clean
    assert changed.any()
    assert torch.all(model.bin_indices[changed] == model._blind_bin())
    # Untouched cells are bit-identical to the clean matrix.
    assert torch.equal(
        model.bin_indices[~changed], model.bin_indices_clean[~changed]
    )


@pytest.mark.parametrize("blind_ratio", [0.05, 0.25, 0.6])
def test_blinded_fraction_matches_blind_ratio_without_importance(blind_ratio):
    """No importance recorded yet means a uniform blind_ratio, as in the reference."""
    model = _blinder_model(
        num_samples=20000, feature_blinder=True, blind_ratio=blind_ratio, blind_seed=3
    )

    model._maybe_blind_cells(round_idx=1)

    fractions = _blinded_fraction_per_feature(model)
    np.testing.assert_allclose(fractions, blind_ratio, atol=0.02)


def test_blinding_probability_scales_with_normalized_importance():
    importance = torch.tensor([8.0, 4.0, 2.0, 0.0], dtype=torch.float32)
    model = _blinder_model(
        num_samples=20000,
        num_features=4,
        importance=importance,
        feature_blinder=True,
        blind_ratio=0.4,
        blind_seed=5,
    )

    model._maybe_blind_cells(round_idx=1)

    fractions = _blinded_fraction_per_feature(model)
    # blind_ratio * (importance / importance.max())
    np.testing.assert_allclose(fractions, [0.4, 0.2, 0.1, 0.0], atol=0.02)


def test_blinding_never_compounds_across_redraws():
    model = _blinder_model(
        feature_blinder=True,
        blind_ratio=0.3,
        blind_seed=17,
    )

    model._maybe_blind_cells(round_idx=1)
    first = _blinded_fraction_per_feature(model).mean()

    model._maybe_blind_cells(round_idx=2)
    second = _blinded_fraction_per_feature(model).mean()

    # Compounding would push the second redraw toward 1 - (1 - 0.3)**2 = 0.51.
    np.testing.assert_allclose([first, second], 0.3, atol=0.02)
    assert model.blind_redraws_ == 2


def test_redraws_produce_different_masks():
    model = _blinder_model(
        feature_blinder=True, blind_ratio=0.3, blind_seed=23
    )

    model._maybe_blind_cells(round_idx=1)
    first = model.bin_indices.clone()

    model._maybe_blind_cells(round_idx=2)

    assert not torch.equal(first, model.bin_indices)


def test_redraw_only_fires_on_the_cadence():
    model = _blinder_model(
        feature_blinder=True,
        blind_ratio=0.3,
        blind_every_n_trees=5,
        blind_seed=29,
    )

    for round_idx in range(5):
        assert model._maybe_blind_cells(round_idx) is False
        assert torch.equal(model.bin_indices, model.bin_indices_clean)

    assert model._maybe_blind_cells(round_idx=5) is True
    after_event = model.bin_indices.clone()

    for round_idx in range(6, 10):
        assert model._maybe_blind_cells(round_idx) is False
        assert torch.equal(model.bin_indices, after_event)

    assert model._maybe_blind_cells(round_idx=10) is True
    assert model.blind_redraws_ == 2


def test_inactive_blinder_leaves_matrix_untouched():
    model = _blinder_model(feature_blinder=False, blind_ratio=0.5)

    assert model.bin_indices_clean is None
    assert model._maybe_blind_cells(round_idx=1) is False


def test_blind_frequency_misses_train_on_clean_data():
    model = _blinder_model(
        feature_blinder=True,
        blind_ratio=0.3,
        blind_frequency=0.5,
        blind_seed=101,
    )

    blinded_rounds = 0
    clean_rounds = 0
    for round_idx in range(1, 41):
        model._maybe_blind_cells(round_idx)
        if torch.equal(model.bin_indices, model.bin_indices_clean):
            clean_rounds += 1
        else:
            blinded_rounds += 1

    assert blinded_rounds > 0, "blind_frequency=0.5 should blind on some events"
    assert clean_rounds > 0, "blind_frequency=0.5 should skip on some events"
    assert model.blind_redraws_ == blinded_rounds


def test_blind_frequency_one_always_blinds():
    model = _blinder_model(
        feature_blinder=True,
        blind_ratio=0.3,
        blind_frequency=1.0,
        blind_seed=7,
    )

    assert model._maybe_blind_cells(0) is False  # warm-up round is never blinded
    for round_idx in range(1, 7):
        assert model._maybe_blind_cells(round_idx) is True

    assert model.blind_redraws_ == 6


def test_blind_seed_is_reproducible():
    def blinded(blind_seed):
        model = _blinder_model(
            feature_blinder=True, blind_ratio=0.3, blind_seed=blind_seed
        )
        model._maybe_blind_cells(round_idx=1)
        return model.bin_indices.clone()

    assert torch.equal(blinded(42), blinded(42))
    assert not torch.equal(blinded(42), blinded(43))


def test_blind_seed_falls_back_to_random_state():
    def blinded(**kwargs):
        model = _blinder_model(feature_blinder=True, blind_ratio=0.3, **kwargs)
        model._maybe_blind_cells(round_idx=1)
        return model.bin_indices.clone()

    assert torch.equal(blinded(random_state=13), blinded(blind_seed=13))
    # An explicit blind_seed wins over random_state.
    assert not torch.equal(
        blinded(random_state=13), blinded(random_state=13, blind_seed=99)
    )


def test_blinding_chunks_cover_every_row():
    """A feature matrix wider than the chunk budget must still blind all rows."""
    from warpgbm import core

    original = core.BLIND_RANDOM_CHUNK_ELEMENTS
    core.BLIND_RANDOM_CHUNK_ELEMENTS = 64  # forces many small chunks
    try:
        model = _blinder_model(
            num_samples=1000,
            num_features=8,
            feature_blinder=True,
            blind_ratio=0.5,
            blind_seed=31,
        )
        model._maybe_blind_cells(round_idx=1)
    finally:
        core.BLIND_RANDOM_CHUNK_ELEMENTS = original

    blinded_rows = (model.bin_indices != model.bin_indices_clean).any(dim=1)
    # With p=0.5 over 8 features, a fully skipped chunk would show as a run of
    # untouched rows; every 8-row window should contain at least one blinded row.
    assert blinded_rows.to(torch.float32).mean().item() > 0.9


# ─────────────────────────── feature selection is colsample-only ───────────────


def test_feature_selection_is_independent_of_blind_seed():
    def selected(random_state, blind_seed):
        torch.manual_seed(random_state)
        model = WarpGBM(
            device="cpu",
            colsample_bytree=0.5,
            random_state=random_state,
            feature_blinder=True,
            blind_ratio=0.3,
            blind_seed=blind_seed,
        )
        model.num_features = 10
        model.feature_indices = torch.arange(10, device=model.device, dtype=torch.int32)
        return model._select_tree_features().detach().cpu().numpy()

    np.testing.assert_array_equal(selected(123, 7), selected(123, 99))

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(selected(123, 7), selected(999, 7))


def test_full_colsample_keeps_the_identity_feature_mapping():
    model = WarpGBM(
        device="cpu", colsample_bytree=1.0, feature_blinder=True, blind_ratio=0.5
    )
    model.num_features = 6
    model.feature_indices = torch.arange(6, device=model.device, dtype=torch.int32)

    selected = model._select_tree_features()

    assert selected is model.feature_indices
    assert model._feat_indices_are_identity is True


def test_identity_feature_mapping_does_not_read_feature_tensor():
    class ExplodingFeatureTensor:
        def __getitem__(self, _):
            raise AssertionError("identity mapping should not index feat_indices_tree")

    model = WarpGBM(device="cpu")
    model._feat_indices_are_identity = True
    model.feat_indices_tree = ExplodingFeatureTensor()

    assert model._global_feature_for_local(3) == 3


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


# ─────────────────────────── root hessian cache invalidation ───────────────────


def test_redraw_invalidates_the_cached_root_hessian():
    model = WarpGBM(
        device="cpu",
        n_estimators=6,
        num_bins=6,
        colsample_bytree=1.0,
        feature_blinder=True,
        blind_ratio=0.5,
        blind_every_n_trees=3,
        blind_seed=7,
    )
    model.num_samples = 32
    model.num_features = 6
    model.num_eras = 1
    model.feature_indices = torch.arange(6, device=model.device, dtype=torch.int32)
    model.root_node_indices = torch.arange(32, device=model.device, dtype=torch.int32)
    model.Y_gpu = torch.ones(32, device=model.device, dtype=torch.float32)
    model.gradients = torch.zeros(32, device=model.device, dtype=torch.float32)
    model.era_sorted_for_hist = True
    model.max_root_era_count = 32
    model.forest = [{} for _ in range(model.n_estimators)]
    model.bin_indices = _clean_matrix(32, 6, 6, device=model.device)
    model._setup_blinder_buffers()

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
    model.grow_tree = lambda *args, **kwargs: {
        "leaf_value": torch.tensor(0.0),
        "samples": 32,
    }
    model.compute_eval = lambda _: None
    model._resync_train_scores = lambda: None

    model.grow_forest()

    # Redraws at rounds 0 and 3 each force a full (grad + hess) recompute;
    # the two rounds after each redraw reuse the cached hessian.
    assert full_histogram_calls == 2
    assert root_grad_only_calls == 4


# ─────────────────────────── training integration (CUDA) ───────────────────────


def _regression_data(n=4000, f=12, seed=0):
    X, y = make_regression(n_samples=n, n_features=f, noise=0.1, random_state=seed)
    return X.astype(np.float32), y.astype(np.float32)


def test_regression_fit_perturbs_cells_and_resyncs_scores():
    X, y = _regression_data()
    seen = []

    model = WarpGBM(
        max_depth=3,
        n_estimators=12,
        num_bins=10,
        feature_blinder=True,
        blind_ratio=0.4,
        blind_every_n_trees=4,
        blind_seed=42,
        random_state=42,
    )

    original_resync = model._resync_train_scores

    def recording_resync():
        original_resync()
        if model._trees_trained > 0:
            expected = model.predict_binned(model.bin_indices)
            seen.append(torch.allclose(model.gradients, expected, atol=1e-5))

    model._resync_train_scores = recording_resync
    model.fit(X, y)

    assert model.blind_redraws_ == 2  # rounds 4 and 8; round 0 is the warm-up
    assert seen and all(seen), "scores must track the perturbed matrix after a redraw"

    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    assert np.all(np.isfinite(preds))

    features = _tree_features(model.forest[: model._trees_trained])
    assert features
    assert all(0 <= int(f) < X.shape[1] for f in features)


def test_regression_blinding_changes_the_fitted_forest():
    X, y = _regression_data()
    common = dict(max_depth=3, n_estimators=15, num_bins=10, random_state=7)

    plain = WarpGBM(**common).fit(X, y)
    blinded = WarpGBM(
        feature_blinder=True,
        blind_ratio=0.5,
        blind_every_n_trees=5,
        blind_seed=7,
        **common,
    ).fit(X, y)

    assert blinded.blind_redraws_ == 2  # rounds 5 and 10
    assert not np.allclose(plain.predict(X), blinded.predict(X))


def test_disabled_blinder_matches_baseline_training():
    X, y = _regression_data()
    common = dict(max_depth=3, n_estimators=10, num_bins=10, random_state=5)

    baseline = WarpGBM(**common).fit(X, y)
    disabled = WarpGBM(
        feature_blinder=False, blind_ratio=0.5, blind_every_n_trees=2, **common
    ).fit(X, y)

    assert disabled.blind_redraws_ == 0
    np.testing.assert_allclose(baseline.predict(X), disabled.predict(X), atol=1e-5)


def test_training_matrices_are_released_after_fit():
    X, y = _regression_data(n=1000, f=8)

    model = WarpGBM(
        n_estimators=4,
        feature_blinder=True,
        blind_ratio=0.3,
        blind_every_n_trees=2,
        blind_seed=1,
    ).fit(X, y)

    assert model.bin_indices_clean is None
    assert not hasattr(model, "bin_indices")


def test_eval_matrix_is_not_blinded():
    X, y = _regression_data(n=2000, f=8)
    X_eval, y_eval = _regression_data(n=500, f=8, seed=1)

    model = WarpGBM(
        n_estimators=6,
        feature_blinder=True,
        blind_ratio=0.5,
        blind_every_n_trees=2,
        blind_seed=3,
    )
    model.fit(X, y, X_eval=X_eval, y_eval=y_eval, eval_every_n_trees=2)

    expected = model.bin_inference_data(X_eval)
    assert torch.equal(model.bin_indices_eval, expected)


def test_multiclass_fit_with_blinder():
    X, y = make_classification(
        n_samples=3000,
        n_features=12,
        n_informative=8,
        n_classes=3,
        random_state=0,
    )
    X = X.astype(np.float32)

    model = WarpGBM(
        objective="multiclass",
        max_depth=3,
        n_estimators=9,
        num_bins=10,
        feature_blinder=True,
        blind_ratio=0.4,
        blind_every_n_trees=3,
        blind_seed=11,
        random_state=11,
    ).fit(X, y)

    assert model.blind_redraws_ == 2  # rounds 3 and 6
    probs = model.predict_proba(X)
    assert probs.shape == (X.shape[0], 3)
    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    features = _tree_features(model.forest)
    assert features
    assert all(0 <= int(f) < X.shape[1] for f in features)


def test_multiclass_round_shares_one_candidate_set():
    X, y = make_classification(
        n_samples=1500,
        n_features=10,
        n_informative=6,
        n_classes=3,
        random_state=2,
    )
    X = X.astype(np.float32)

    model = WarpGBM(
        objective="multiclass",
        n_estimators=3,
        colsample_bytree=0.5,
        feature_blinder=True,
        blind_ratio=0.4,
        blind_every_n_trees=1,
        blind_seed=2,
        random_state=2,
    )

    per_round_selections = []
    original_select = model._select_tree_features

    def recording_select():
        selected = original_select()
        per_round_selections.append(selected.detach().cpu().numpy().copy())
        return selected

    model._select_tree_features = recording_select
    model.fit(X, y)

    # One draw per boosting round, not per class tree.
    assert len(per_round_selections) == model._trees_trained


def test_warm_start_continues_training_with_blinder():
    X, y = _regression_data(n=2000, f=10)

    model = WarpGBM(
        n_estimators=6,
        warm_start=True,
        feature_blinder=True,
        blind_ratio=0.3,
        blind_every_n_trees=3,
        blind_seed=4,
        random_state=4,
    )
    model.fit(X, y)
    assert model._trees_trained == 6

    model.n_estimators = 12
    model.fit(X, y)

    assert model._trees_trained == 12
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_fit_is_reproducible_with_a_fixed_blind_seed():
    X, y = _regression_data(n=2000, f=10)

    def fitted():
        return WarpGBM(
            n_estimators=8,
            feature_blinder=True,
            blind_ratio=0.4,
            blind_every_n_trees=2,
            blind_seed=99,
            random_state=99,
        ).fit(X, y).predict(X)

    np.testing.assert_allclose(fitted(), fitted(), atol=1e-5)


# ─────────────────────────── persistence ───────────────────────────


def test_save_load_preserves_blinder_parameters(tmp_path):
    X, y = _regression_data(n=1000, f=8)

    model = WarpGBM(
        n_estimators=4,
        feature_blinder=True,
        blind_ratio=0.35,
        blind_frequency=0.7,
        blind_every_n_trees=2,
        blind_seed=21,
    ).fit(X, y)

    path = tmp_path / "model.pkl"
    model.save_model(str(path))

    loaded = WarpGBM().load_model(str(path))

    assert loaded.feature_blinder is True
    assert loaded.blind_ratio == 0.35
    assert loaded.blind_frequency == 0.7
    assert loaded.blind_every_n_trees == 2
    assert loaded.blind_seed == 21
    np.testing.assert_allclose(model.predict(X), loaded.predict(X), atol=1e-5)


def test_load_defaults_blinder_parameters_for_legacy_state(tmp_path):
    X, y = _regression_data(n=1000, f=8)
    model = WarpGBM(n_estimators=4).fit(X, y)

    path = tmp_path / "legacy.pkl"
    model.save_model(str(path))

    with open(path, "rb") as handle:
        state = pickle.load(handle)
    for key in ("feature_blinder", "blind_ratio", "blind_frequency",
                "blind_every_n_trees", "blind_seed"):
        state.pop(key, None)
    with open(path, "wb") as handle:
        pickle.dump(state, handle)

    loaded = WarpGBM().load_model(str(path))

    assert loaded.feature_blinder is False
    assert loaded.blind_ratio == 0.0
    assert loaded.blind_frequency == 1.0
    assert loaded.blind_every_n_trees == 50
    assert loaded.blind_seed is None
