"""All project configurations and hyperparameters"""


BASE_DIR = "scibert/"
CKPTH_DIR = "checkpoints/"
DATA_DIR = "data/raw"
DATA = "data/raw/20230821_full_data.pkl"
TEST_DIR = "data/raw/test_ids.xlsx"
TRAIN_PROCESSED_DATA = "data/processed/train.pkl"
VAL_PROCESSED_DATA = "data/processed/val.pkl"
TEST_PROCESSED_DATA = "data/processed/test.pkl"


LOG_DIR = BASE_DIR + "/logs"

LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> >> <level>{message}</level>"


LABEL_MAPPER = {"Relevant": 1, "Not relevant": 0}
VAL_SIZE = 0.2

MODEL = "tinyBERT"

SEED = 123
BATCH_SIZE = 32
EPOCHS = 50
TOKENS_MAX_LENGTH = 512
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


def concatenation_strategy(x):
    return "[SEP]".join(x)
