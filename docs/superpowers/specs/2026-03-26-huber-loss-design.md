# Huber Loss Design

**Date:** 2026-03-26
**Status:** Approved

## Summary

Add Huber loss as a training option for regression in WarpGBM, exposed via a new `loss` sub-parameter. Uses gradient clipping (clipped-residual approximation) to avoid any CUDA kernel changes.

## API

Two new constructor parameters on `WarpGBM`:

```python
loss="mse",        # "mse" | "huber"
huber_delta=1.0,   # threshold; only used when loss="huber"
```

- `loss` applies only to `objective="regression"`. Passing `loss="huber"` with a classification objective raises `ValueError`.
- `huber_delta` must be a positive float. Default is `1.0`.

## Gradient Computation

The existing regression residual computation in `grow_forest`:

```python
self.residual = self.Y_gpu - self.gradients
```

Becomes:

```python
residual = self.Y_gpu - self.gradients
if self.loss == "huber":
    self.residual = torch.clamp(residual, -self.huber_delta, self.huber_delta)
else:
    self.residual = residual
```

This is the clipped-gradient approximation of Huber. The CUDA histogram kernel hardcodes `hessian = 1` per sample — which is slightly wrong for true Huber (outlier hessian should be 0) but is a well-known practical approximation that requires no CUDA changes.

## Eval Metric

Add `"huber"` to valid `eval_metric` values. Implementation in `get_eval_metric`:

```python
elif self.eval_metric == "huber":
    delta = self.huber_delta
    r = y_true - y_pred
    loss = torch.where(r.abs() <= delta, 0.5 * r ** 2, delta * (r.abs() - 0.5 * delta))
    return loss.mean().item()
```

Available as an eval metric regardless of training loss — users can monitor Huber loss while training with MSE, and vice versa.

## Validation

- `_validate_hyperparams`: add `"huber"` to valid `loss` values; check `huber_delta > 0`
- `fit()`: raise `ValueError` if `loss="huber"` and `objective` is not `"regression"`
- `validate_fit_params`: add `"huber"` to `valid_metrics`

## Files Changed

- `warpgbm/core.py` — all changes (no CUDA files touched)

## Out of Scope

- True Huber with `hessian=0` for outliers (requires CUDA kernel change)
- Pseudo-Huber / smooth Huber variant
- Auto-tuning of `huber_delta` from data
