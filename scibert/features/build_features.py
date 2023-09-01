from scibert.preprocessing.make_data import make
from scibert.config import DATA, TEST_DIR


def combine_features(df):
    assert (
        all(x in df.columns for x in ["title", "keywords", "content", "target"]) == True
    )

    # create a single sentence that combines the entire three columns as X
    # and target as y

    X = df[["title", "keywords", "content"]].apply(lambda x: "[SEP]".join(x), axis=1)

    y = df["target"]

    return X.values, y.values


def build_features(train, test):
    train_X, train_y = combine_features(train)
    test_X, test_y = combine_features(test)

    return train_X, train_y, test_X, test_y


if __name__ == "__main__":
    # import training and testing dataframes
    train, test = make(DATA, TEST_DIR)
