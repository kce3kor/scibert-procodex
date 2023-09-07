import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scibert.config import DATA, TEST_DIR, VAL_SIZE, concatenation_strategy
from scibert.preprocessing.make_data import make
from scibert.utils.logger import logger


def combine_features(df: pd.DataFrame) -> np.ndarray:
    """Given a dataframe, combine the respective columns to form a single sequence,
       Combines the sequences with [SEP]

    Args:
        df (pd.DataFrame): Input Dataframe (must contain ["title", "keywords", "content", "target"])

    Returns:
        np.ndarray: len X 1 : Returns a combined sequence delimited by [SEP] for input to model
    """
    assert all(x in df.columns for x in ["title", "keywords", "content", "target"]) == True

    # create a single sentence that combines the entire three columns as X
    # and target as y

    X = df[["title", "keywords", "content"]].apply(concatenation_strategy, axis=1)

    y = df["target"]

    return X.values, y.values


def zero_rule_baseline(l):
    counts = {}
    for x in l:
        if x not in counts:
            counts[x] = 1
        else:
            counts[x] += 1

    max_label = max(counts, key=lambda k: counts.get(k))
    return counts, counts[max_label] / len(l)


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Build training and testing features from train and test dataframes individually,
    Features are combined text from individual columns for the given dataframes

    Args:
        train (pd.DataFrame): Training Data
        test (pd.DataFrame): Testing Data

    Returns:
        np.ndarray: train_X, train_y, test_X, test_y (numpy arrays)
    """
    logger.info("Building features with each columns: String concatenation delimited with [SEP]")
    X, y = combine_features(train)

    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, stratify=y)

    test_X, test_y = combine_features(test)

    print(f"Train: {train_X.shape, train_y.shape}, Distribution: {zero_rule_baseline(train_y)}")

    print(f"Validation: {val_X.shape, val_y.shape}, Distribution: {zero_rule_baseline(val_y)}")
    print(f"Test: {test_X.shape, test_y.shape}, Distribution: {zero_rule_baseline(test_y)}")

    return train_X, train_y, val_X, val_y, test_X, test_y


if __name__ == "__main__":
    # import training and testing dataframes
    train, test = make(DATA, TEST_DIR)
    train_X, train_y, val_X, val_y, test_X, test_y = build_features(train, test)
