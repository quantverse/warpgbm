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
