import re
import string
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os, requests
from pathlib import Path

from scibert.utils.logger import logger


# 1. lower case
def lower_case(text):
    return text.lower()


# 2. remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


# 3. remove stop words
def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_text)


# 4. remove numbers
def remove_numbers(text):
    return re.sub(r"\d+", "", text)


# 5. remove short words
def remove_short_words(text):
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if len(w) > 2]
    return " ".join(filtered_text)


# 6. lemmatize
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(w) for w in word_tokens]
    return " ".join(lemmatized_text)


# 8. remove non-ascii characters
def remove_non_ascii(text):
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


# 9. remove extra spaces
def remove_extra_spaces(text):
    return re.sub(" +", " ", text)


# 16. remove extra non-breaking spaces
def remove_extra_non_breaking_spaces(text):
    text = re.sub("\xa0+", "", text)
    text = re.sub("\x0c", "", text)

    return text


def remove_dashes_underscores(text):
    text = text.replace("-", "")
    text = text.replace("_", "")

    return text


def remove_special_characters(text):
    # Define a regular expression pattern to match special characters
    pattern = r"[^a-zA-Z0-9\s]"  # This pattern matches anything that's not alphanumeric or whitespace

    # Use re.sub() to replace matched patterns with an empty string
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text


def remove_urls(text):
    # Define a regular expression pattern to match URLs
    pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

    # Use re.sub() to replace matched URLs with an empty string
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text


from scibert.utils.decorators import timer


class DataFetcher:
    def __init__(self, files):
        self.files = files

    @staticmethod
    def download(file):
        # Exception catcher for individual files
        try:
            r = requests.get(file["url"], stream=True)
            if not os.path.exists(file["destination"]):
                os.makedirs(file["destination"])

            with open(Path(file["destination"], file["filename"]), "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())

        except Exception as e:
            logger.error(f"Download failed for file: {file['url']}::{e}")

    @timer
    def fetch(self):
        for file in self.files:
            logger.info(f"Downloading file: {file['filename']}")
            self.download(file)
