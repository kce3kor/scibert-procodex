import pytest
import pandas as pd
from scibert.utils.serializer import pickle_serializer
from scibert.config import DATA, TEST_DIR
from scibert.preprocessing.make_data import make


@pytest.mark.data
def test_data_columns():
    data = pickle_serializer(path=DATA, mode="load")
    test = pd.read_excel(TEST_DIR)

    assert "id" in test.columns
    assert "id" in data.columns

    assert len(data.columns) == 5

    assert set(data.columns) == set(["id", "title", "keywords", "content", "target"])


@pytest.mark.data
def test_make_data():
    train, test = make(DATA, TEST_DIR)

    test_raw = pd.read_excel(TEST_DIR)

    assert train.shape == (1595, 5)
    assert test.shape == (786, 5)

    assert len(test) == len(test_raw)
