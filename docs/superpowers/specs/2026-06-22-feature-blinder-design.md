# Feature Blinder for Training Design

> **SUPERSEDED (2026-08-06).** The candidate-blinding design described below was replaced by
> per-cell perturbation of the binned training matrix, matching `LightGbmBlinderTrainer`.
> In particular the Context section's rejection of cell perturbation no longer holds.
> See `BLINDER.md` for the implemented design.

## Context

WarpGBM trains histogram-based trees on pre-binned feature tensors. The current training loop already supports per-tree feature candidate selection through `colsample_bytree`: each tree receives `feat_indices_tree`, split search operates on local feature positions, and selected splits are stored as global feature ids. This makes feature-candidate blinding a natural training-time extension.

The reference `LightGbmBlinderTrainer` in `~/Projects/atol/monorepo` bootstraps a LightGBM model, then repeatedly trains chunks on copies of the training matrix where individual cells are reset to a neutral value. Cell perturbation would force WarpGBM to mutate or re-bin feature data during training. The selected WarpGBM design instead blinds features from split candidates, preserving the CUDA histogram flow.

## Goal

Add an optional training feature that discourages repeated dependence on currently important features by randomly excluding a fraction of candidate features from each boosting round. The feature should work for regression, binary classification, multiclass classification, era-aware training, and warm start.

## Non-Goals

This design does not add LightGBM-compatible cell perturbation, per-cell neutral value replacement, new prediction-time behavior, or a separate model class. It does not change the tree representation used for saved models or prediction.

## Public API

Add constructor parameters to `WarpGBM`:

```python
feature_blinder=False
blind_ratio=0.0
blind_frequency=1.0
blind_seed=None
```

`feature_blinder` enables the feature. `blind_ratio` is the fraction of current candidate features to blind when a blinding event occurs. `blind_frequency` is the per-tree probability of applying blinding. `blind_seed` optionally gives the blinder its own reproducible seed; when omitted, `random_state` is used for deterministic blinder draws if available.

Validation rules:

- `feature_blinder` must be a boolean.
- `blind_ratio` must be a float in `[0.0, 1.0)`.
- `blind_frequency` must be a float in `[0.0, 1.0]`.
- `blind_seed` must be `None` or an integer.
- A blinding event always leaves at least one feature candidate.

## Selected Algorithm

Each boosting round builds `feat_indices_tree` through a helper, for example `_select_tree_features(round_idx)`.

1. Start from all global features.
2. Apply existing `colsample_bytree` logic to choose a per-tree candidate set.
3. If `feature_blinder` is enabled and `blind_ratio > 0.0`, draw a Bernoulli event using `blind_frequency`.
4. If the event fires, remove `floor(blind_ratio * candidate_count)` features from the candidate set, capped at `candidate_count - 1`.
5. Choose features to remove with probabilities proportional to current cumulative feature importance.
6. If feature importance is unavailable or all zero, fall back to uniform removal.
7. Train the tree using the remaining candidate set.

The selected split still maps from local candidate position to global feature id before storage, exactly as current `colsample_bytree` handling does.

When actual blinding is disabled (`feature_blinder=False` or `blind_ratio=0.0`), the current candidate-selection behavior is preserved. When actual blinding is enabled and `blind_seed` or `random_state` is set, blinder-specific draws are generated from the round index and seed so a resumed warm-start run does not depend on persisted RNG state.

## Components

`_validate_hyperparams` validates the new constructor arguments.

`_select_tree_features(round_idx)` centralizes feature candidate selection for regression and classification. It replaces duplicated `colsample_bytree` blocks in `grow_forest()` and `grow_forest_multiclass()`.

`_blind_feature_candidates(candidate_features)` handles the importance-weighted removal. It reads from `per_era_feature_importance_gpu` during training so weights remain on the active device, and falls back to uniform sampling when importances are all zero.

The existing `find_best_split()`, histogram kernels, tree representation, prediction code, and feature-importance accumulation remain unchanged.

## Data Flow

`fit()` validates inputs and seeds the existing random streams. Regression and classification setup remains unchanged. At the start of every boosting round, the forest grower calls `_select_tree_features(round_idx)` and receives the candidate tensor to use for root histograms and all descendant node histograms in that tree.

Because blinding affects only candidate selection, the binned training matrix, validation matrix, stored trees, and inference path are unchanged.

## Warm Start and Serialization

Warm start continues from `_trees_trained` as it does today. Existing feature importance from already-trained trees remains available and can guide subsequent blinding. With a fixed `blind_seed` or `random_state`, blinder event and removal draws are deterministic per round. Existing `colsample_bytree` behavior remains unchanged when actual blinding is disabled.

`save_model()` and `load_model()` persist and restore the new hyperparameters. Models saved before this feature can be loaded by using defaults when keys are missing.

## Error Handling

Invalid parameter types or ranges fail during construction with clear exceptions. If blinding removes every feature after rounding or tiny candidate counts, the implementation clamps the removal count so the tree still has at least one feature. No special handling is needed for no-split trees; existing leaf fallback behavior remains valid.

## Testing

Add focused tests for:

- Constructor validation for `feature_blinder`, `blind_ratio`, `blind_frequency`, and `blind_seed`.
- Feature blinder leaves at least one candidate when `blind_ratio` is high.
- With `blind_ratio=0.0` or `feature_blinder=False`, results match the current behavior.
- With fixed `random_state`, repeated blinder training is reproducible.
- With different seeds or random states, blinder training can produce different trees.
- Save/load preserves new blinder parameters.
- Warm start with blinder can continue training without shape or state errors.

For behavioral verification, use small synthetic regression datasets and inspect trained tree feature ids or prediction equality where appropriate. Avoid large benchmark-style tests in the unit suite.

## Risks

The main risk is making warm-start reproducibility diverge from current expectations if blinder random draws are not aligned between uninterrupted and resumed training. The implementation should derive seeded blinder draws from the boosting-round index when `blind_seed` or `random_state` is available, and keep candidate selection in one helper so regression and classification do not drift.

The second risk is feature starvation when both `colsample_bytree` and `blind_ratio` are aggressive. Clamping removal count to preserve one candidate avoids crashes but can weaken the intended regularization on very small candidate sets.
