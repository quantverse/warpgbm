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
