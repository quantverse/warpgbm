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
        ({"blind_ratio": False}, TypeError),
        ({"blind_ratio": True}, TypeError),
        ({"blind_frequency": -0.01}, ValueError),
        ({"blind_frequency": 1.01}, ValueError),
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
