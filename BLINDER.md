# Feature Blinder Implementation

This branch implements feature blinding as a native WarpGBM training-time regularizer. It is a port of `LightGbmBlinderTrainer.perturb_data_per_cell`, applied directly to the GPU binned matrix rather than by wrapping LightGBM.

Blinding operates **per cell of the training matrix**. Individual `(row, feature)` cells are overwritten with an uninformative value, so no feature can be counted on to always be observable. Feature-level candidate dropping is *not* what this does — the split-candidate set is controlled by `colsample_bytree` alone.

## Public API

`WarpGBM` accepts five blinder parameters:

```python
WarpGBM(
    feature_blinder=False,
    blind_ratio=0.0,
    blind_frequency=1.0,
    blind_every_n_trees=50,
    blind_seed=None,
)
```

- `feature_blinder`: enables per-cell blinding when `True`.
- `blind_ratio`: base per-cell blinding probability — the `cell_prob` of the reference. Must be `>= 0.0` and `< 1.0`.
- `blind_frequency`: probability that a redraw event actually blinds. On a miss, training runs on clean data until the next event. Must be `>= 0.0` and `<= 1.0`.
- `blind_every_n_trees`: how often the mask is redrawn, in trees. The analogue of the reference's `BLIND_STEPS = 1000`. Must be `>= 1`.
- `blind_seed`: optional independent seed for blinder draws. If omitted, `random_state` is used when available. `colsample_bytree` subsampling remains controlled by `random_state` and PyTorch's active RNG state, and is entirely independent of the blinder.

The blinder is active only when all of these hold:

```python
feature_blinder is True
blind_ratio > 0.0
blind_frequency > 0.0
```

Validation rejects invalid types, booleans passed as numeric blinder parameters, non-finite ratios/frequencies, out-of-range values, non-positive `blind_every_n_trees`, and non-integer `blind_seed` values.

## Perturbation Algorithm

The implementation lives in `warpgbm/core.py`:

- `_feature_blinder_active()`
- `_setup_blinder_buffers()` / `_release_blinder_buffers()`
- `_blind_bin()`
- `_blind_cell_probabilities()`
- `_maybe_blind_cells(round_idx)`
- `_resync_train_scores()` / `_multiclass_train_scores(bin_indices)`

### Buffers

When the blinder is active, `fit` keeps the pristine binned matrix as `bin_indices_clean` and hands training a working clone in `bin_indices`. The extra `int8` `n_samples × n_features` allocation exists only while the blinder is on and is released at the end of `fit`.

### Redraw

`_maybe_blind_cells(round_idx)` runs at the top of each boosting round and returns early unless:

```python
round_idx > 0 and round_idx % blind_every_n_trees == 0
```

Round 0 is deliberately never blinded. The reference trains a first block of `BLIND_STEPS` rounds on clean data before its first call to `perturb_data_per_cell`, so that every perturbation is weighted by importance the model has actually accumulated. Skipping round 0 reproduces that: the first `blind_every_n_trees` trees fit clean data.

When the round qualifies:

1. A `torch.Generator` is seeded from `blind_seed` (falling back to `random_state`) strided by `round_idx`.
2. The `blind_frequency` event is drawn. On a miss the matrix is restored to clean, training scores are resynced if it had been blinded, and the function returns.
3. Per-feature probabilities are computed exactly as in the reference:

   ```python
   fimp = per_era_feature_importance_gpu.sum(dim=0).clamp_min(0.0)
   fimp = fimp / fimp.max()          # ones(F) if max == 0
   feature_probs = blind_ratio * fimp
   ```

   Importance must be read from `per_era_feature_importance_gpu`. On this branch the split-importance accumulation is a CUDA kernel that writes only to that tensor; the `per_era_feature_importance_` numpy mirror is synced once, after the boosting loop, and is all zeros during training. Reading the mirror would silently disable blinding.

   The all-ones fallback mirrors the reference's `if fimp is None` branch. Combined with the round-0 skip it is only reached when the forest has recorded no splits at all.
4. The working matrix is reset from `bin_indices_clean` and the mask is applied in row chunks:

   ```python
   draws = torch.rand((chunk_rows, num_features), generator=generator)
   bin_indices[start:stop].masked_fill_(draws < feature_probs, blind_bin)
   ```

   Chunking is what keeps this usable at scale. A full-matrix `torch.rand` would need `4 × n_samples × n_features` bytes of scratch — roughly 16 GB on a 2M × 2000 Numerai matrix. `BLIND_RANDOM_CHUNK_ELEMENTS` caps the random buffer at about 256 MB and the fill is in place, so peak memory stays flat in the number of features.

Every redraw starts from `bin_indices_clean`, so perturbations never compound: two consecutive redraws at `blind_ratio=0.3` leave 30% of cells blinded, not 51%.

### Blind Value

A blinded cell is set to the middle bin, `(num_bins - 1) // 2`. Under quantile binning that is the median bin, the closest binned-space analogue of the reference's `value=0.5` on `[0, 1]` features.

### Score Resync

Blinded cells reroute rows through the already-grown trees, so the running prediction state goes stale after a redraw. `_resync_train_scores()` rebuilds it against the current matrix — the equivalent of LightGBM warm-starting a booster on a freshly built `Dataset`:

- Regression: `gradients = predict_binned(bin_indices)`.
- Multiclass and binary: `gradients = _multiclass_train_scores(bin_indices)`, which is the training class priors `log(prior_k)` plus every stored tree re-predicted on the current matrix.

Without this, residuals would be computed against a matrix the histograms are no longer built from.

`_multiclass_train_scores` also serves the initial score setup in `_fit_classification`, which replaced a hand-rolled prior loop plus a per-tree Python traversal.

### Root Hessian Cache

`grow_forest` caches the root hessian histogram across trees when eras are sorted and `colsample_bytree == 1.0`. Bin counts move when cells move, so a redraw invalidates that cache:

```python
if self._maybe_blind_cells(i):
    root_hessian_histogram_cache = None
```

## Training Integration

Regression (`grow_forest`) and multiclass/binary (`grow_forest_multiclass`) both call `_maybe_blind_cells(i)` at the top of each boosting round, before residuals or softmax gradients are derived from the current scores.

Per-tree feature selection is now `colsample_bytree` only, in `_select_tree_features()`. Multiclass draws that candidate set once per boosting round and shares it across the round's class trees.

Only the training matrix is perturbed. `bin_indices_eval` is built once from `X_eval` and never touched, matching the reference's static validation set. Trained trees store global feature ids and ordinary bin thresholds, so prediction is unaffected by whether blinding was used.

Warm-start resumption re-preprocesses `X` into a clean matrix and rebuilds scores from the stored forest, then re-enters the same global `round_idx % blind_every_n_trees` schedule.

## Cost

A redraw costs one pass over the full matrix plus one full-forest re-prediction. The re-prediction makes total training cost grow with the square of the number of redraws, so `blind_every_n_trees=1` is expensive on large data. The default of 50 keeps it marginal.

## Determinism

With `blind_seed` set, the mask for a given round is deterministic. The per-round seed is:

```python
(seed + round_idx * BLIND_SEED_STRIDE) % 2**64
```

Seeded blinder draws do not make the whole CUDA training run bitwise deterministic; the histogram and split kernels retain the nondeterminism documented elsewhere for WarpGBM.

## Relationship to the Reference

| `LightGbmBlinderTrainer` | This implementation |
| --- | --- |
| `cell_prob` | `blind_ratio` |
| `BLIND_STEPS = 1000` | `blind_every_n_trees` |
| `fimp / fimp.max()`, `np.ones` when absent | identical, on `per_era_feature_importance_gpu` |
| `value=0.5` on `[0, 1]` features | middle bin, `(num_bins - 1) // 2` |
| `X_mod` always derived from `x` | every redraw resets from `bin_indices_clean` |
| clean rounds before the first perturbation | round 0 skipped |
| warm start on a new `Dataset` | `_resync_train_scores()` |
| static `dm_val` | `bin_indices_eval` untouched |

`blind_frequency` has no counterpart in `LightGbmBlinderTrainer`; it carries over the `doblind` draw from `ann_utils.gen_batch_blinder_sampling`.

The reference's outer early-stopping loop over `EX_TOTAL_ITERS` is not ported. WarpGBM already has `eval_every_n_trees` / `early_stopping_rounds`, which cover the same ground.
