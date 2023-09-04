from scibert.preprocessing.make_data import make
from scibert.config import DATA, TEST_DIR
from scibert.utils.logger import logger

import pandas as pd
import numpy as np


def combine_features(df: pd.DataFrame) -> np.ndarray:
    """Given a dataframe, combine the respective columns to form a single sequence,
       Combines the sequences with [SEP]

    Args:
        df (pd.DataFrame): Input Dataframe (must contain ["title", "keywords", "content", "target"])

    Returns:
        np.ndarray: len X 1 : Returns a combined sequence delimited by [SEP] for input to model
    """
    assert (
        all(x in df.columns for x in ["title", "keywords", "content", "target"]) == True
    )

    # create a single sentence that combines the entire three columns as X
    # and target as y

    X = df[["title", "keywords", "content"]].apply(lambda x: "[SEP]".join(x), axis=1)

    y = df["target"]

    return X.values, y.values


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Build training and testing features from train and test dataframes individually,
    Features are combined text from individual columns for the given dataframes

    Args:
        train (pd.DataFrame): Training Data
        test (pd.DataFrame): Testing Data

    Returns:
        np.ndarray: train_X, train_y, test_X, test_y (numpy arrays)
    """
    logger.info(
        "Building features with each columns: String concatenation delimited with [SEP]"
    )
    train_X, train_y = combine_features(train)
    test_X, test_y = combine_features(test)

    return train_X, train_y, test_X, test_y


if __name__ == "__main__":
    # import training and testing dataframes
    train, test = make(DATA, TEST_DIR)
