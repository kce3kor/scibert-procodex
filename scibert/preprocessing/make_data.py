import os, itertools

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from scibert.utils.serializer import pickle_serializer
from scibert.preprocessing.utils import (
    remove_punctuation,
    lemmatize,
    remove_extra_non_breaking_spaces,
    remove_non_ascii,
    remove_dashes_underscores,
    remove_special_characters,
    remove_extra_spaces,
    remove_urls,
    DataFetcher,
)

from scibert.config import DATA_DIR, LABEL_MAPPER
from scibert.utils.logger import logger


def process_target(label: str) -> int:
    """Returns target labels mapped to int

    Args:
        label (str): Target in str

    Returns:
        int: Target in int
    """
    return LABEL_MAPPER[label]


def process_title(text: str) -> str:
    """Preprocessing Title Column in the Dataset

    Args:
        text (str): Title for all data rows

    Returns:
        str: processed title for each rows
    """
    text = remove_punctuation(text)
    text = lemmatize(text)
    text = remove_extra_non_breaking_spaces(text)
    return text


def process_keywords(text: str) -> str:
    """Preprocessing keywords in data

    Args:
        text (str): keywords from dataframe

    Returns:
        str: processed keywords for each data rows
    """
    tokens = text.split(";")
    tokens = [remove_non_ascii(t) for t in tokens]
    tokens = [t.split("/") for t in tokens]
    tokens = list(itertools.chain.from_iterable(tokens))
    tokens = [remove_extra_non_breaking_spaces(t) for t in tokens]
    tokens = [lemmatize(t) for t in tokens]
    return " ".join(tokens)


def process_content(text: str) -> str:
    """Processed Content for each data rows

    Args:
        text (str): Raw Contents

    Returns:
        str: Processed Content for each rows
    """
    x = text.replace("\n", " ")
    x = remove_urls(x)
    x = remove_non_ascii(x)
    x = remove_extra_non_breaking_spaces(x)
    x = remove_dashes_underscores(x)
    x = remove_special_characters(x)
    x = remove_extra_spaces(x)
    return x


def preprocess(df: pd.DataFrame, preprocesses: dict) -> pd.DataFrame:
    """Preprocess each columns with respective processing functions

    Args:
        df (pd.DataFrame): Data
        preprocesses (dict): Respective processing functions based on each columns

    Returns:
        pd.DataFrame: Processed Data
    """
    logger.info(f"Preprocessing Dataframe for each columns")
    for column, transform in tqdm(preprocesses.items()):
        df[column] = df[column].apply(transform)

    return df


def make(data_path: str, test_path: str) -> pd.DataFrame:
    """Make processed train and test data based on raw dataframe of complete data and test ids

    Args:
        data_path (str): Path to original data
        test_path (str): Path to test ids

    Raises:
        Exception: Raises missing column expection if the required columns are not there

    Returns:
        pd.DataFrame: train and test dataframe after preprocessing
    """
    # LOAD ENVIRONMENT VARIABLES
    logger.info(f"Loading environment variables")
    _ = load_dotenv()

    files = [
        {
            "url": os.getenv("DATASET_LOC"),
            "destination": DATA_DIR,
            "filename": "data.csv",
        }
    ]
    # DOWNLOAD REQUIRED DATA
    # DataFetcher(files).fetch()

    # PREPROCESSING STARTS
    data = pickle_serializer(path=data_path, mode="load")
    test = test = pd.read_excel(test_path)

    # PROCESSED DATA
    pdata = preprocess(
        data,
        preprocesses={
            "title": process_title,
            "keywords": process_keywords,
            "content": process_content,
            "target": process_target,
        },
    )

    if "id" not in test.columns or "id" not in pdata.columns:
        raise Exception("Missing 'id' column in dataframes")

    test_ids = list(test["id"].values)

    logger.info(f"Creating training and testing sets")
    train = pdata[~pdata["id"].isin(test_ids)]
    test = pdata[pdata["id"].isin(test_ids)]

    return train, test


if __name__ == "__main__":
    from scibert.config import DATA, TEST_DIR

    train, test = make(DATA, TEST_DIR)
    print(train.shape, test.shape)
