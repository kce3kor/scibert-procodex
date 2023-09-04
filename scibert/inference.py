import random
import numpy as np
import torch

from pathlib import Path

from scibert.preprocessing.make_data import (
    process_content,
    process_title,
    process_keywords,
)
from scibert.models.dispatcher import MODELS
from scibert.utils.logger import logger

from scibert.config import MODEL, TOKENS_MAX_LENGTH, CKPTH_DIR, SEED, LABEL_MAPPER


def initialize(seed):
    logger.info(f"Initializing development pipeline with SEED: {seed}")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.derterministic = True
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")

    return device


def load_model(ckpth):
    logger.info("Loading saved model from: {ckpth}")
    model = MODELS[MODEL]["model"]

    model.load_state_dict(torch.load(ckpth))

    return model


def reverse_label_mapper(label):
    return list(LABEL_MAPPER.keys())[list(LABEL_MAPPER.values()).index(label)]


def inference(query):
    logger.info(f"Query: {query}")

    device = initialize(SEED)

    query = {
        "title": process_title(query["title"]),
        "content": process_content(query["content"]),
        "keywords": process_keywords(query["keywords"]),
    }

    input = query["title"] + "[SEP]" + query["keywords"] + "[SEP]" + query["content"]
    logger.info(f"Model input: {input}")

    tokenizer = MODELS[MODEL]["tokenizer"]

    inputs = tokenizer.encode_plus(
        input,
        add_special_tokens=True,
        max_length=TOKENS_MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    ).to(device)

    model = load_model(Path(CKPTH_DIR, f"{MODEL}_{9}.pt"))
    model = model.to(device)

    logger.info("Running model evaluation...")
    model.eval()

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
    logits = logits.detach().cpu().numpy()

    label = np.argmax(logits, axis=1)
    prediction = reverse_label_mapper(label)

    logger.info(f"Output: {logits, label, prediction}")

    return prediction


if __name__ == "__main__":
    query = {
        "title": "This is the title",
        "content": "This is the content",
        "keywords": "This is a keywords",
    }
    print(inference(query))
