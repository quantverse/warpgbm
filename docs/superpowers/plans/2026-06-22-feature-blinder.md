# Feature Blinder Implementation Plan

> **SUPERSEDED (2026-08-06).** This plan built per-tree candidate blinding, which was
> replaced by per-cell matrix perturbation. See `BLINDER.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional WarpGBM-native feature blinding that removes high-importance features from per-tree split candidates during training.

**Architecture:** Add constructor parameters and validation to `WarpGBM`, centralize feature candidate selection in helper methods, and call that helper from both regression and multiclass forest growth. Blinding operates only on training-time feature candidates; tree storage, prediction, and CUDA kernels remain unchanged.

**Tech Stack:** Python, PyTorch tensors and generators, NumPy, pytest, existing WarpGBM CUDA extension.

---

## File Structure

- Modify `warpgbm/core.py`: public API parameters, validation, feature-candidate helper methods, dynamic split-buffer sizing, regression and multiclass integration, save/load persistence.
- Create `tests/test_feature_blinder.py`: focused tests for validation, helper behavior, reproducibility, integration, warm start, and serialization.
- Modify `README.md`: document the new constructor parameters and a short usage example.

## Task 1: Public API and Validation

**Files:**
- Create: `tests/test_feature_blinder.py`
- Modify: `warpgbm/core.py:17-149`

- [ ] **Step 1: Write failing validation tests**

Create `tests/test_feature_blinder.py` with these imports and tests:

```python
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
        ({"blind_frequency": -0.01}, ValueError),
        ({"blind_frequency": 1.01}, ValueError),
        ({"blind_frequency": "1.0"}, TypeError),
        ({"blind_seed": 1.5}, TypeError),
        ({"blind_seed": True}, TypeError),
    ],
)
def test_feature_blinder_constructor_validation(kwargs, expected_error):
    with pytest.raises(expected_error):
        WarpGBM(**kwargs)
```

- [ ] **Step 2: Run validation tests and verify they fail**

Run:

```bash
rtk pytest tests/test_feature_blinder.py::test_feature_blinder_constructor_defaults tests/test_feature_blinder.py::test_feature_blinder_constructor_validation -q
```

Expected: FAIL because `WarpGBM.__init__()` does not accept `feature_blinder`, `blind_ratio`, `blind_frequency`, or `blind_seed`.

- [ ] **Step 3: Add constructor parameters and validation**

In `warpgbm/core.py`, update `WarpGBM.__init__` signature to include the new parameters after `warm_start`:

```python
        warm_start=False,
        feature_blinder=False,
        blind_ratio=0.0,
        blind_frequency=1.0,
        blind_seed=None,
```

Pass them to `_validate_hyperparams`:

```python
            colsample_bytree=colsample_bytree,
            feature_blinder=feature_blinder,
            blind_ratio=blind_ratio,
            blind_frequency=blind_frequency,
            blind_seed=blind_seed,
```

Assign the attributes after `self.warm_start = warm_start`:

```python
        self.feature_blinder = feature_blinder
        self.blind_ratio = blind_ratio
        self.blind_frequency = blind_frequency
        self.blind_seed = blind_seed
```

Add `blind_ratio` and `blind_frequency` to `float_params`:

```python
        float_params = [
            "learning_rate",
            "min_split_gain",
            "L2_reg",
            "colsample_bytree",
            "blind_ratio",
            "blind_frequency",
        ]
```

After the `float_params` loop, add bool and seed validation:

```python
        if not isinstance(kwargs["feature_blinder"], bool):
            raise TypeError(
                f"feature_blinder must be a boolean, got {type(kwargs['feature_blinder'])}."
            )
        if kwargs["blind_seed"] is not None:
            if isinstance(kwargs["blind_seed"], bool) or not isinstance(
                kwargs["blind_seed"], int
            ):
                raise TypeError(
                    f"blind_seed must be None or an integer, got {type(kwargs['blind_seed'])}."
                )
```

After `colsample_bytree` range validation, add:

```python
        if kwargs["blind_ratio"] < 0.0 or kwargs["blind_ratio"] >= 1.0:
            raise ValueError(
                f"Invalid blind_ratio: {kwargs['blind_ratio']}. Must be >= 0.0 and < 1.0."
            )
        if kwargs["blind_frequency"] < 0.0 or kwargs["blind_frequency"] > 1.0:
            raise ValueError(
                f"Invalid blind_frequency: {kwargs['blind_frequency']}. Must be >= 0.0 and <= 1.0."
            )
```

- [ ] **Step 4: Run validation tests and verify they pass**

Run:

```bash
rtk pytest tests/test_feature_blinder.py::test_feature_blinder_constructor_defaults tests/test_feature_blinder.py::test_feature_blinder_constructor_validation -q
```

Expected: PASS for both tests.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add warpgbm/core.py tests/test_feature_blinder.py
rtk git commit -m "feat: add feature blinder parameters"
```

Expected: commit succeeds.

## Task 2: Candidate Selection Helpers

**Files:**
- Modify: `tests/test_feature_blinder.py`
- Modify: `warpgbm/core.py:151-220`

- [ ] **Step 1: Add failing helper tests**

Append these tests to `tests/test_feature_blinder.py`:

```python
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
```

- [ ] **Step 2: Run helper tests and verify they fail**

Run:

```bash
rtk pytest tests/test_feature_blinder.py::test_feature_blinder_candidate_selection_keeps_one_feature tests/test_feature_blinder.py::test_feature_blinder_prefers_blinding_important_feature tests/test_feature_blinder.py::test_feature_blinder_frequency_zero_preserves_candidates tests/test_feature_blinder.py::test_feature_blinder_round_seed_is_reproducible -q
```

Expected: FAIL because `_select_tree_features` does not exist.

- [ ] **Step 3: Add feature selection helper methods**

Add these methods after `_compute_softmax_gradients_hessians` in `warpgbm/core.py`:

```python
    def _feature_blinder_active(self):
        return (
            self.feature_blinder
            and self.blind_ratio > 0.0
            and self.blind_frequency > 0.0
        )

    def _feature_selection_generator(self, round_idx):
        seed = self.blind_seed if self.blind_seed is not None else self.random_state
        if seed is None or not self._feature_blinder_active():
            return None

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed) + int(round_idx) * 1_000_003)
        return generator

    def _select_tree_features(self, round_idx):
        generator = self._feature_selection_generator(round_idx)
        generator_kwargs = {"generator": generator} if generator is not None else {}

        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
            self.feat_indices_tree = torch.randperm(
                self.num_features,
                device=self.device,
                dtype=torch.int32,
                **generator_kwargs,
            )[:k]
        else:
            self.feat_indices_tree = self.feature_indices

        self.feat_indices_tree = self._blind_feature_candidates(
            self.feat_indices_tree, generator=generator
        )
        return self.feat_indices_tree

    def _blind_feature_candidates(self, candidate_features, generator=None):
        if not self._feature_blinder_active():
            return candidate_features

        generator_kwargs = {"generator": generator} if generator is not None else {}

        candidate_count = int(candidate_features.numel())
        if candidate_count <= 1:
            return candidate_features

        if self.blind_frequency < 1.0:
            event = torch.rand((), device=self.device, **generator_kwargs).item()
            if event >= self.blind_frequency:
                return candidate_features

        blind_count = min(
            int(np.floor(self.blind_ratio * candidate_count)),
            candidate_count - 1,
        )
        if blind_count <= 0:
            return candidate_features

        weights = None
        if hasattr(self, "per_era_feature_importance_gpu"):
            weights = self.per_era_feature_importance_gpu.sum(dim=0).to(torch.float32)
            weights = weights[candidate_features.to(torch.long)].clamp_min(0.0)

        if weights is None or weights.sum().item() <= 0:
            remove_local = torch.randperm(
                candidate_count,
                device=self.device,
                **generator_kwargs,
            )[:blind_count]
        else:
            remove_local = torch.multinomial(
                weights + 1e-12,
                num_samples=blind_count,
                replacement=False,
                **generator_kwargs,
            )

        keep_mask = torch.ones(candidate_count, dtype=torch.bool, device=self.device)
        keep_mask[remove_local] = False
        return candidate_features[keep_mask].contiguous()

    def _prepare_split_buffers(self, candidate_count):
        split_bins = self.num_bins - 1
        if self.num_eras > 1:
            self.fused_gain = torch.zeros(
                candidate_count,
                split_bins,
                device=self.device,
                dtype=torch.float32,
            )
            self.fused_agreement = torch.zeros_like(self.fused_gain)
        else:
            self.per_era_gain = torch.zeros(
                1,
                candidate_count,
                split_bins,
                device=self.device,
                dtype=torch.float32,
            )
            self.per_era_direction = torch.zeros_like(self.per_era_gain)
```

- [ ] **Step 4: Run helper tests and verify they pass**

Run:

```bash
rtk pytest tests/test_feature_blinder.py::test_feature_blinder_candidate_selection_keeps_one_feature tests/test_feature_blinder.py::test_feature_blinder_prefers_blinding_important_feature tests/test_feature_blinder.py::test_feature_blinder_frequency_zero_preserves_candidates tests/test_feature_blinder.py::test_feature_blinder_round_seed_is_reproducible -q
```

Expected: PASS for all helper tests.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add warpgbm/core.py tests/test_feature_blinder.py
rtk git commit -m "feat: add feature blinder candidate selection"
```

Expected: commit succeeds.

## Task 3: Regression Forest Integration

**Files:**
- Modify: `tests/test_feature_blinder.py`
- Modify: `warpgbm/core.py:861-945`

- [ ] **Step 1: Add failing regression integration tests**

Append these tests to `tests/test_feature_blinder.py`:

```python
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
```

- [ ] **Step 2: Run regression integration tests and verify they fail**

Run:

```bash
rtk pytest tests/test_feature_blinder.py::test_feature_blinder_regression_fit_uses_blinded_candidates tests/test_feature_blinder.py::test_feature_blinder_disabled_matches_existing_regression_behavior -q
```

Expected: FAIL because `grow_forest()` still sizes split buffers before blinding and never calls `_select_tree_features(round_idx)`.

- [ ] **Step 3: Integrate helper in `grow_forest()`**

In `warpgbm/core.py`, remove the pre-loop `k` calculation and pre-loop split-buffer allocation from `grow_forest()`:

```python
        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            self.feat_indices_tree = self.feature_indices
            k = self.num_features
            
        if self.num_eras > 1:
            self.fused_gain = torch.zeros(k, self.num_bins-1, device=self.device, dtype=torch.float32)
            self.fused_agreement = torch.zeros(k, self.num_bins-1, device=self.device, dtype=torch.float32)
        else:
            self.per_era_gain = torch.zeros(1, k, self.num_bins-1, device=self.device, dtype=torch.float32)
            self.per_era_direction = torch.zeros(1, k, self.num_bins-1, device=self.device, dtype=torch.float32)
```

Replace `can_reuse_root_hessian` with:

```python
        can_reuse_root_hessian = (
            self.era_sorted_for_hist
            and self.colsample_bytree >= 1.0
            and not self._feature_blinder_active()
            and self.max_root_era_count > 0
        )
```

Inside the `for i in range(start_iter, self.n_estimators):` loop, replace:

```python
            if self.colsample_bytree < 1.0:
                self.feat_indices_tree = torch.randperm(self.num_features, device=self.device, dtype=torch.int32)[:k]
```

with:

```python
            self.feat_indices_tree = self._select_tree_features(round_idx=i)
            self._prepare_split_buffers(int(self.feat_indices_tree.numel()))
```

- [ ] **Step 4: Run regression integration tests and verify they pass**

Run:

```bash
rtk pytest tests/test_feature_blinder.py::test_feature_blinder_regression_fit_uses_blinded_candidates tests/test_feature_blinder.py::test_feature_blinder_disabled_matches_existing_regression_behavior -q
```

Expected: PASS for both tests.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add warpgbm/core.py tests/test_feature_blinder.py
rtk git commit -m "feat: use feature blinder in regression training"
```

Expected: commit succeeds.

## Task 4: Multiclass Integration, Warm Start, and Serialization

**Files:**
- Modify: `tests/test_feature_blinder.py`
- Modify: `warpgbm/core.py:947-1037`
- Modify: `warpgbm/core.py:1322-1432`

- [ ] **Step 1: Add failing classification, warm-start, and save/load tests**

Append these tests to `tests/test_feature_blinder.py`:

```python
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
        num_bins=16,
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
        num_bins=16,
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
        num_bins=16,
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
        num_bins=16,
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
```

- [ ] **Step 2: Run new tests and verify they fail**

Run:

```bash
rtk pytest tests/test_feature_blinder.py::test_feature_blinder_multiclass_fit_runs tests/test_feature_blinder.py::test_feature_blinder_warm_start_continues_training tests/test_feature_blinder.py::test_feature_blinder_reproducible_with_fixed_seed tests/test_feature_blinder.py::test_feature_blinder_save_load_preserves_parameters tests/test_feature_blinder.py::test_feature_blinder_loads_legacy_state_defaults -q
```

Expected: FAIL because multiclass still uses old candidate selection and save/load does not persist the new parameters.

- [ ] **Step 3: Integrate helper in `grow_forest_multiclass()`**

In `warpgbm/core.py`, remove the pre-loop `k` calculation and pre-loop split-buffer allocation from `grow_forest_multiclass()`:

```python
        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            self.feat_indices_tree = self.feature_indices
            k = self.num_features
            
        if self.num_eras > 1:
            self.fused_gain = torch.zeros(k, self.num_bins-1, device=self.device, dtype=torch.float32)
            self.fused_agreement = torch.zeros(k, self.num_bins-1, device=self.device, dtype=torch.float32)
        else:
            self.per_era_gain = torch.zeros(1, k, self.num_bins-1, device=self.device, dtype=torch.float32)
            self.per_era_direction = torch.zeros(1, k, self.num_bins-1, device=self.device, dtype=torch.float32)
```

Inside the `for i in range(start_iter, self.n_estimators):` loop, replace:

```python
            if self.colsample_bytree < 1.0:
                self.feat_indices_tree = torch.randperm(self.num_features, device=self.device, dtype=torch.int32)[:k]
```

with:

```python
            self.feat_indices_tree = self._select_tree_features(round_idx=i)
            self._prepare_split_buffers(int(self.feat_indices_tree.numel()))
```

Keep this selection before the `for class_k in range(self.num_classes):` loop so every class tree in the boosting round uses the same feature candidates.

- [ ] **Step 4: Persist blinder parameters in save/load**

In `save_model()`, add these keys after `'warm_start': self.warm_start,`:

```python
            'feature_blinder': self.feature_blinder,
            'blind_ratio': self.blind_ratio,
            'blind_frequency': self.blind_frequency,
            'blind_seed': self.blind_seed,
```

In `load_model()`, add these assignments after `self.warm_start = state['warm_start']`:

```python
        self.feature_blinder = state.get('feature_blinder', False)
        self.blind_ratio = state.get('blind_ratio', 0.0)
        self.blind_frequency = state.get('blind_frequency', 1.0)
        self.blind_seed = state.get('blind_seed', None)
```

- [ ] **Step 5: Run Task 4 tests and verify they pass**

Run:

```bash
rtk pytest tests/test_feature_blinder.py::test_feature_blinder_multiclass_fit_runs tests/test_feature_blinder.py::test_feature_blinder_warm_start_continues_training tests/test_feature_blinder.py::test_feature_blinder_reproducible_with_fixed_seed tests/test_feature_blinder.py::test_feature_blinder_save_load_preserves_parameters tests/test_feature_blinder.py::test_feature_blinder_loads_legacy_state_defaults -q
```

Expected: PASS for all five tests.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add warpgbm/core.py tests/test_feature_blinder.py
rtk git commit -m "feat: support feature blinder training flows"
```

Expected: commit succeeds.

## Task 5: Documentation and Full Verification

**Files:**
- Modify: `README.md:340-410`

- [ ] **Step 1: Update README constructor reference**

In `README.md`, update the constructor example to include the new parameters after `warm_start=False`:

```python
    warm_start=False,              # If True, continue training from existing trees
    feature_blinder=False,         # If True, blind feature candidates during training
    blind_ratio=0.0,               # Fraction of candidate features to blind per event
    blind_frequency=1.0,           # Probability of applying blinding each tree
    blind_seed=None,               # Optional seed for deterministic blinder draws
```

- [ ] **Step 2: Add README usage example**

Add this short section after the feature-importance utility example:

````markdown
### Feature Blinder Training

Feature blinding is a training-time regularizer that randomly removes a fraction of current split candidates, weighted toward features that already have high cumulative importance:

```python
model = WarpGBM(
    n_estimators=200,
    max_depth=4,
    feature_blinder=True,
    blind_ratio=0.4,
    blind_frequency=0.8,
    blind_seed=42,
)
model.fit(X_train, y_train, era_id=era_id)
```

Prediction is unchanged because trained trees still store global feature ids.
````

- [ ] **Step 3: Run feature blinder test file**

Run:

```bash
rtk pytest tests/test_feature_blinder.py -q
```

Expected: PASS for the full feature blinder test file.

- [ ] **Step 4: Run focused existing regression, classification, and warm-start tests**

Run:

```bash
rtk pytest tests/test_random_state.py tests/test_warm_start.py::TestWarmStartRegression::test_warm_start_equivalent_to_full_training tests/test_warm_start.py::TestWarmStartClassification::test_multiclass_warm_start_equivalent -q
```

Expected: PASS for the selected existing tests.

- [ ] **Step 5: Run status check**

Run:

```bash
rtk git status --short
```

Expected: only intentional changes in `README.md`, `tests/test_feature_blinder.py`, and `warpgbm/core.py`.

- [ ] **Step 6: Commit documentation and verification-ready state**

Run:

```bash
rtk git add README.md tests/test_feature_blinder.py warpgbm/core.py
rtk git commit -m "docs: document feature blinder training"
```

Expected: commit succeeds.

## Self-Review Notes

Spec coverage:

- Public API parameters are covered in Task 1.
- Constructor validation is covered in Task 1.
- Importance-weighted candidate removal and all-zero fallback are covered in Task 2.
- Regression integration is covered in Task 3.
- Multiclass integration is covered in Task 4.
- Warm start and deterministic seed behavior are covered in Task 4.
- Save/load compatibility is covered in Task 4.
- README documentation is covered in Task 5.

Type consistency:

- Helper names are consistently `_feature_blinder_active`, `_feature_selection_generator`, `_select_tree_features`, `_blind_feature_candidates`, and `_prepare_split_buffers`.
- Public attributes are consistently `feature_blinder`, `blind_ratio`, `blind_frequency`, and `blind_seed`.
- Tests use `device="cuda"` because `WarpGBM.__init__` currently allocates CUDA tensors by default and the existing test suite is GPU-oriented.
