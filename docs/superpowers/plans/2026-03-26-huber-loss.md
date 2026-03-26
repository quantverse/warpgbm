# Huber Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Huber loss as a training option for regression via a `loss` sub-parameter on `WarpGBM`, using gradient clipping with no CUDA changes.

**Architecture:** Add `loss="mse"|"huber"` and `huber_delta=1.0` parameters to `WarpGBM.__init__`. In `grow_forest`, clamp the residual to `[-huber_delta, huber_delta]` when `loss="huber"`. Add `"huber"` as a valid `eval_metric` using the standard Huber loss formula.

**Tech Stack:** PyTorch (GPU tensors), sklearn-compatible API, pytest

---

## File Map

- Modify: `warpgbm/core.py` — all changes live here
- Create: `tests/test_huber_loss.py` — all new tests

---

### Task 1: Parameter declaration and validation

**Files:**
- Create: `tests/test_huber_loss.py`
- Modify: `warpgbm/core.py:15-145`

- [ ] **Step 1: Write failing tests for parameter validation**

Create `tests/test_huber_loss.py`:

```python
import pytest
import numpy as np
from warpgbm import WarpGBM


def _make_regression_data(n=500, f=20, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, f).astype(np.float32)
    y = (X @ rng.randn(f)).astype(np.float32)
    return X, y


def test_huber_default_params():
    model = WarpGBM(loss="huber")
    assert model.loss == "huber"
    assert model.huber_delta == 1.0


def test_mse_is_default_loss():
    model = WarpGBM()
    assert model.loss == "mse"


def test_invalid_loss_raises():
    with pytest.raises(ValueError, match="loss must be"):
        WarpGBM(loss="invalid")


def test_invalid_huber_delta_raises():
    with pytest.raises(ValueError, match="huber_delta must be"):
        WarpGBM(loss="huber", huber_delta=-1.0)


def test_huber_delta_zero_raises():
    with pytest.raises(ValueError, match="huber_delta must be"):
        WarpGBM(loss="huber", huber_delta=0.0)


def test_huber_custom_delta():
    model = WarpGBM(loss="huber", huber_delta=2.5)
    assert model.huber_delta == 2.5
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_huber_loss.py::test_huber_default_params tests/test_huber_loss.py::test_mse_is_default_loss tests/test_huber_loss.py::test_invalid_loss_raises tests/test_huber_loss.py::test_invalid_huber_delta_raises tests/test_huber_loss.py::test_huber_delta_zero_raises tests/test_huber_loss.py::test_huber_custom_delta -v
```

Expected: All FAIL (TypeError or AttributeError — `loss` param doesn't exist yet).

- [ ] **Step 3: Add `loss` and `huber_delta` to `__init__`**

In `warpgbm/core.py`, change the `__init__` signature (lines 15–31) from:

```python
    def __init__(
        self,
        objective="regression",
        num_bins=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=20,
        min_split_gain=0.0,
        threads_per_block=64,
        rows_per_thread=4,
        L2_reg=1e-6,
        device="cuda",
        colsample_bytree=1.0,
        random_state=None,
        warm_start=False,
    ):
```

to:

```python
    def __init__(
        self,
        objective="regression",
        num_bins=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=20,
        min_split_gain=0.0,
        threads_per_block=64,
        rows_per_thread=4,
        L2_reg=1e-6,
        device="cuda",
        colsample_bytree=1.0,
        random_state=None,
        warm_start=False,
        loss="mse",
        huber_delta=1.0,
    ):
```

- [ ] **Step 4: Pass `loss` and `huber_delta` to `_validate_hyperparams` and assign them**

In `__init__`, change the `_validate_hyperparams` call (lines 33–45) from:

```python
        # Validate arguments
        self._validate_hyperparams(
            objective=objective,
            num_bins=num_bins,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            threads_per_block=threads_per_block,
            rows_per_thread=rows_per_thread,
            L2_reg=L2_reg,
            colsample_bytree=colsample_bytree,
        )
```

to:

```python
        # Validate arguments
        self._validate_hyperparams(
            objective=objective,
            num_bins=num_bins,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            threads_per_block=threads_per_block,
            rows_per_thread=rows_per_thread,
            L2_reg=L2_reg,
            colsample_bytree=colsample_bytree,
            loss=loss,
            huber_delta=huber_delta,
        )
```

Then add the assignments at the end of `__init__` (after `self._trees_trained = 0`):

```python
        self.loss = loss
        self.huber_delta = huber_delta
```

- [ ] **Step 5: Add validation rules to `_validate_hyperparams`**

In `_validate_hyperparams` (lines 83–143), add after the `colsample_bytree` check (after line 143):

```python
        if kwargs["loss"] not in ["mse", "huber"]:
            raise ValueError(
                f"loss must be 'mse' or 'huber', got {kwargs['loss']}."
            )
        if not isinstance(kwargs["huber_delta"], (int, float)) or kwargs["huber_delta"] <= 0:
            raise ValueError(
                f"huber_delta must be a positive number, got {kwargs['huber_delta']}."
            )
```

- [ ] **Step 6: Run tests to verify they pass**

```
pytest tests/test_huber_loss.py::test_huber_default_params tests/test_huber_loss.py::test_mse_is_default_loss tests/test_huber_loss.py::test_invalid_loss_raises tests/test_huber_loss.py::test_invalid_huber_delta_raises tests/test_huber_loss.py::test_huber_delta_zero_raises tests/test_huber_loss.py::test_huber_custom_delta -v
```

Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add warpgbm/core.py tests/test_huber_loss.py
git commit -m "feat: add loss and huber_delta parameters with validation"
```

---

### Task 2: Reject `loss="huber"` with classification objectives

**Files:**
- Modify: `tests/test_huber_loss.py`
- Modify: `warpgbm/core.py` (the `fit()` method, ~line 293)

- [ ] **Step 1: Write failing test**

Append to `tests/test_huber_loss.py`:

```python
def test_huber_with_binary_raises():
    X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    model = WarpGBM(objective="binary", loss="huber", n_estimators=2)
    with pytest.raises(ValueError, match="loss='huber' is only supported"):
        model.fit(X, y)


def test_huber_with_multiclass_raises():
    X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 2, 0])
    model = WarpGBM(objective="multiclass", loss="huber", n_estimators=2)
    with pytest.raises(ValueError, match="loss='huber' is only supported"):
        model.fit(X, y)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_huber_loss.py::test_huber_with_binary_raises tests/test_huber_loss.py::test_huber_with_multiclass_raises -v
```

Expected: Both FAIL (no error is raised yet).

- [ ] **Step 3: Add objective/loss cross-validation to `fit()`**

In `warpgbm/core.py`, in the `fit()` method (around line 340, after the warm-start block and before the era_id default), add:

```python
        if self.loss == "huber" and self.objective != "regression":
            raise ValueError(
                f"loss='huber' is only supported for objective='regression', "
                f"got objective='{self.objective}'."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_huber_loss.py::test_huber_with_binary_raises tests/test_huber_loss.py::test_huber_with_multiclass_raises -v
```

Expected: Both PASS.

- [ ] **Step 5: Commit**

```bash
git add warpgbm/core.py tests/test_huber_loss.py
git commit -m "feat: reject loss='huber' with classification objectives"
```

---

### Task 3: Huber gradient computation in `grow_forest`

**Files:**
- Modify: `tests/test_huber_loss.py`
- Modify: `warpgbm/core.py` (`grow_forest`, ~line 719)

- [ ] **Step 1: Write failing test**

Append to `tests/test_huber_loss.py`:

```python
def test_huber_trains_and_predicts():
    """Model trains with loss='huber' and produces reasonable predictions."""
    X, y = _make_regression_data()
    model = WarpGBM(
        objective="regression",
        loss="huber",
        huber_delta=1.0,
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)
    preds = model.predict(X)
    corr = np.corrcoef(preds, y)[0, 1]
    assert corr > 0.8, f"In-sample correlation too low with Huber loss: {corr:.4f}"


def test_huber_more_robust_to_outliers():
    """Huber model should be less influenced by outliers than MSE."""
    rng = np.random.RandomState(0)
    X = rng.randn(400, 10).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)

    # Inject severe outliers into training set only
    y_train = y.copy()
    outlier_idx = rng.choice(400, size=20, replace=False)
    y_train[outlier_idx] += 50.0

    # Clean test targets
    X_test = rng.randn(100, 10).astype(np.float32)
    y_test = (X_test[:, 0] * 2 + X_test[:, 1]).astype(np.float32)

    params = dict(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42)

    model_mse = WarpGBM(objective="regression", loss="mse", **params)
    model_mse.fit(X, y_train)
    mse_preds = model_mse.predict(X_test)

    model_huber = WarpGBM(objective="regression", loss="huber", huber_delta=1.0, **params)
    model_huber.fit(X, y_train)
    huber_preds = model_huber.predict(X_test)

    mse_error = np.mean((mse_preds - y_test) ** 2)
    huber_error = np.mean((huber_preds - y_test) ** 2)

    # Huber should have lower test MSE than MSE model trained on outlier data
    assert huber_error < mse_error, (
        f"Huber model (MSE={huber_error:.4f}) should outperform MSE model "
        f"(MSE={mse_error:.4f}) on clean test set when trained with outliers."
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_huber_loss.py::test_huber_trains_and_predicts tests/test_huber_loss.py::test_huber_more_robust_to_outliers -v
```

Expected: Both FAIL (Huber gradient clipping not implemented yet — model trains as MSE).

- [ ] **Step 3: Clip residuals in `grow_forest`**

In `warpgbm/core.py`, in `grow_forest`, change line 719 from:

```python
            self.residual = self.Y_gpu - self.gradients
```

to:

```python
            residual = self.Y_gpu - self.gradients
            if self.loss == "huber":
                self.residual = torch.clamp(residual, -self.huber_delta, self.huber_delta)
            else:
                self.residual = residual
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_huber_loss.py::test_huber_trains_and_predicts tests/test_huber_loss.py::test_huber_more_robust_to_outliers -v
```

Expected: Both PASS.

- [ ] **Step 5: Commit**

```bash
git add warpgbm/core.py tests/test_huber_loss.py
git commit -m "feat: implement Huber gradient clipping in grow_forest"
```

---

### Task 4: Huber eval metric

**Files:**
- Modify: `tests/test_huber_loss.py`
- Modify: `warpgbm/core.py` (`validate_fit_params` ~line 285, `get_eval_metric` ~line 661)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_huber_loss.py`:

```python
def test_huber_eval_metric_accepted():
    """eval_metric='huber' should be accepted and produce eval loss values."""
    X, y = _make_regression_data()
    model = WarpGBM(
        objective="regression",
        loss="huber",
        huber_delta=1.0,
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    model.fit(X, y, X_eval=X, y_eval=y, eval_every_n_trees=5, eval_metric="huber")
    assert len(model.eval_loss) > 0, "eval_loss should be populated"
    assert all(v >= 0 for v in model.eval_loss), "Huber loss values must be non-negative"


def test_huber_eval_metric_on_mse_model():
    """eval_metric='huber' works even when training with loss='mse'."""
    X, y = _make_regression_data()
    model = WarpGBM(
        objective="regression",
        loss="mse",
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    model.fit(X, y, X_eval=X, y_eval=y, eval_every_n_trees=5, eval_metric="huber")
    assert len(model.eval_loss) > 0


def test_invalid_eval_metric_raises():
    """eval_metric='unknown' should raise ValueError."""
    X, y = _make_regression_data()
    model = WarpGBM(objective="regression", n_estimators=2)
    with pytest.raises(ValueError, match="Invalid eval_metric"):
        model.fit(X, y, X_eval=X, y_eval=y, eval_every_n_trees=1, eval_metric="unknown")
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_huber_loss.py::test_huber_eval_metric_accepted tests/test_huber_loss.py::test_huber_eval_metric_on_mse_model tests/test_huber_loss.py::test_invalid_eval_metric_raises -v
```

Expected: First two FAIL (`"huber"` not in `valid_metrics`). Third may already pass.

- [ ] **Step 3: Add `"huber"` to `valid_metrics` in `validate_fit_params`**

In `warpgbm/core.py`, change (around line 285):

```python
            valid_metrics = ["mse", "corr", "rmsle", "logloss", "accuracy"]
```

to:

```python
            valid_metrics = ["mse", "corr", "rmsle", "logloss", "accuracy", "huber"]
```

- [ ] **Step 4: Add Huber case to `get_eval_metric`**

In `warpgbm/core.py`, in `get_eval_metric` (around line 661), change:

```python
    def get_eval_metric(self, y_true, y_pred):
        if self.eval_metric == "mse":
            return ((y_true - y_pred) ** 2).mean().item()
        elif self.eval_metric == "corr":
            return 1 - torch.corrcoef(torch.vstack([y_true, y_pred]))[0, 1].item()
        elif self.eval_metric == "rmsle":
            return rmsle_torch(y_true, y_pred).item()
        else:
            raise ValueError(f"Invalid eval_metric: {self.eval_metric}.")
```

to:

```python
    def get_eval_metric(self, y_true, y_pred):
        if self.eval_metric == "mse":
            return ((y_true - y_pred) ** 2).mean().item()
        elif self.eval_metric == "corr":
            return 1 - torch.corrcoef(torch.vstack([y_true, y_pred]))[0, 1].item()
        elif self.eval_metric == "rmsle":
            return rmsle_torch(y_true, y_pred).item()
        elif self.eval_metric == "huber":
            delta = self.huber_delta
            r = y_true - y_pred
            loss = torch.where(
                r.abs() <= delta,
                0.5 * r ** 2,
                delta * (r.abs() - 0.5 * delta),
            )
            return loss.mean().item()
        else:
            raise ValueError(f"Invalid eval_metric: {self.eval_metric}.")
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_huber_loss.py::test_huber_eval_metric_accepted tests/test_huber_loss.py::test_huber_eval_metric_on_mse_model tests/test_huber_loss.py::test_invalid_eval_metric_raises -v
```

Expected: All PASS.

- [ ] **Step 6: Run the full test suite**

```
pytest tests/test_huber_loss.py -v
```

Expected: All PASS.

- [ ] **Step 7: Run existing tests to confirm no regressions**

```
pytest tests/test_random_state.py tests/test_fit_predict_corr.py tests/test_warm_start.py -v
```

Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
git add warpgbm/core.py tests/test_huber_loss.py
git commit -m "feat: add huber eval metric"
```
