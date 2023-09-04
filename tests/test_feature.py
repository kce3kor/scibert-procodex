import pytest
import numpy as np


from scibert.config import DATA, TEST_DIR
from scibert.preprocessing.make_data import make
from scibert.features.build_features import build_features


@pytest.mark.features
def test_build_features():
    train, test = make(DATA, TEST_DIR)

    train_X, train_y, test_X, test_y = build_features(train, test)

    assert len(train_X) == len(train_y)
    assert len(test_X) == len(test_y)

    assert type(train_X) == np.ndarray
    assert type(test_X) == np.ndarray

    assert train_X.ndim == 1
    assert test_X.ndim == 1
