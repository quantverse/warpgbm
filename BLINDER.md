# Feature Blinder Implementation

This branch implements feature blinding as a native WarpGBM training-time regularizer. The implementation does not wrap or depend on LightGBM. It applies directly inside the existing GPU tree-building loops by reducing the set of feature candidates considered for each tree.

## Public API

`WarpGBM` accepts four new constructor parameters:

```python
WarpGBM(
    feature_blinder=False,
    blind_ratio=0.0,
    blind_frequency=1.0,
    blind_seed=None,
)
```

Their behavior is:

- `feature_blinder`: enables feature candidate blinding when `True`.
- `blind_ratio`: fraction of the current candidate set to remove. It must be `>= 0.0` and `< 1.0`.
- `blind_frequency`: probability of applying a blinding event for a tree. It must be `>= 0.0` and `<= 1.0`.
- `blind_seed`: optional independent seed for blinder draws. If omitted, `random_state` is used when available. Existing `colsample_bytree` feature-subsampling draws remain controlled by `random_state` and PyTorch's active RNG state.

The blinder is active only when all of these are true:

```python
feature_blinder is True
blind_ratio > 0.0
blind_frequency > 0.0
```

Validation rejects invalid types, booleans passed as numeric blinder parameters, non-finite ratios/frequencies, out-of-range values, and non-integer `blind_seed` values.

## Candidate Selection Algorithm

The core implementation lives in `warpgbm/core.py`:

- `_feature_blinder_active()`
- `_feature_selection_generator(round_idx)`
- `_select_tree_features(round_idx)`
- `_blind_feature_candidates(candidate_features, generator=None)`
- `_prepare_split_buffers(candidate_count)`

For each tree, WarpGBM builds the feature candidate list in two stages:

1. Apply existing `colsample_bytree` logic.
   - If `colsample_bytree < 1.0`, choose a random subset of feature ids.
   - Otherwise, use all feature ids.
2. Optionally blind a subset of those candidates.
   - Skip if the blinder is inactive.
   - Skip if there is only one candidate feature.
   - Draw a blinding event using `blind_frequency`.
   - Compute `blind_count = floor(blind_ratio * candidate_count)`, capped so at least one candidate remains.
   - Remove `blind_count` local candidates.

Removal is importance-weighted when feature importance has signal:

```python
weights = per_era_feature_importance_gpu.sum(dim=0)[candidate_features]
```

The weights are clamped to non-negative values. If the importance tensor is missing or all candidate weights sum to zero, the implementation falls back to uniform random removal. This means early trees can use uniform blinding, and later trees increasingly blind features that have already accumulated high split importance.

The returned candidate tensor is contiguous and always contains at least one feature.

## Determinism

When `blind_seed` is set, blinder draws are deterministic per tree. Otherwise the implementation uses `random_state` when available. The per-tree seed is:

```python
(seed + tree_idx * 1_000_003) % 2**64
```

This keeps blinder selection reproducible while avoiding reuse of the exact same random stream for every tree. If neither `blind_seed` nor `random_state` is set, PyTorch's default generator is used for blinder draws. Seeded blinder draws do not make the entire CUDA training run bitwise deterministic; histogram and split kernels can still have the same nondeterminism noted for the rest of WarpGBM.

Warm-start regression training passes the real tree index into `_select_tree_features`, so continued training uses the same blinder selection schedule it would have used in uninterrupted training.

## Native Training Integration

The feature blinder is integrated into the native WarpGBM loops, not layered on top of training data preprocessing.

Regression training calls `_select_tree_features(round_idx=i)` once per tree in `grow_forest()`, then computes root histograms using the selected feature ids.

Multiclass and binary training draw the `colsample_bytree` candidate set once per boosting iteration in `grow_forest_multiclass()`, preserving the existing shared-column subset across class trees. They then call `_select_tree_features_from_candidates(..., round_idx=i * num_classes + class_k)` for each class tree, so only the blinding step can differ across the per-class trees in a boosting iteration.

The selected tensor is stored as:

```python
self.feat_indices_tree
```

Histogram kernels receive this tensor and score split candidates in local candidate-index space. `grow_tree()` then maps the winning local feature index back to a global feature id before storing the split, splitting rows, and accumulating feature importance:

```python
split_feature = self._global_feature_for_local(local_feature)
```

That mapping is important because the blinder can reduce candidates even when `colsample_bytree == 1.0`. Trained trees still store global feature ids, so prediction code does not need to know whether blinding was used.

The root hessian histogram cache is reused only on rounds whose actual selected candidate set is the full identity feature list. Blinded rounds compute their own root histograms. Split scoring buffers are prepared with `_prepare_split_buffers(candidate_count)`, which keeps enough capacity for the largest candidate set seen and reuses views for smaller blinded sets.

## Persistence

`save_model()` stores the new blinder parameters:

- `feature_blinder`
- `blind_ratio`
- `blind_frequency`
- `blind_seed`

`load_model()` restores those fields and uses legacy defaults when loading older model files that do not contain them:

```python
feature_blinder=False
blind_ratio=0.0
blind_frequency=1.0
blind_seed=None
```

## Behavior Boundaries

Feature blinding is training-only. It removes split candidates during tree construction; it does not remove columns from input data and does not change prediction traversal.

Disabled settings preserve existing behavior. In particular, `feature_blinder=False`, `blind_ratio=0.0`, or `blind_frequency=0.0` all leave candidate selection equivalent to normal WarpGBM feature sampling.

Blinding happens after `colsample_bytree`, so `blind_ratio` is applied to the already-sampled candidate set rather than to the full feature count.

## Tests

Coverage was added in `tests/test_feature_blinder.py` for:

- constructor defaults and validation
- numeric edge cases, including booleans, NaN, and infinities
- candidate selection and one-feature preservation
- importance-weighted removal and uniform fallback behavior
- deterministic selection with fixed seeds
- split-buffer reuse and reallocation
- regression fit integration
- disabled behavior parity
- multiclass fit integration
- warm-start continuation
- save/load and legacy load defaults

Related random-state and warm-start tests were also exercised because the feature changes the per-round candidate selection path used by native training.
