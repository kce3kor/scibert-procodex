import os
import random
from pathlib import Path

import torch
from transformers import AutoModel

from scibert.config import *
from scibert.main import LightningModel
from scibert.models.dispatcher import MODELS
from scibert.preprocessing.make_data import (
    process_content,
    process_keywords,
    process_title,
)
from scibert.utils.logger import logger


def load_model(ckpth: str) -> AutoModel:
    """Model serializer based on the checkpoint

    Args:
        ckpth (str): model checkpoint path

    Returns:
        AutoModel: Transformer Automodel
    """
    logger.info(f"Loading saved model from: {ckpth}")

    model = MODELS[MODEL]["model"]

    lightining_model = LightningModel.load_from_checkpoint(ckpth, model=model)

    return lightining_model


def reverse_label_mapper(label):
    return list(LABEL_MAPPER.keys())[list(LABEL_MAPPER.values()).index(label)]


def inference_preprocess(data, preprocesses):
    preprocessed = {}
    for column, transform in preprocesses.items():
        preprocessed[column] = transform(data[column])

    return preprocessed


def inference(query: dict) -> str:
    """Run inference in the trained model checkpoint and return the model prediction

    Args:
        query (dict): User input respective columns for inference

    Returns:
        str: model prediction as a class value
    """
    logger.info(f"Query: {query}")

    query = inference_preprocess(
        query,
        preprocesses={
            "title": process_title,
            "keywords": process_keywords,
            "content": process_content,
        },
    )

    X = concatenation_strategy(query.values())

    tokenizer = MODELS[MODEL]["tokenizer"]

    inputs = tokenizer.encode_plus(
        X,
        add_special_tokens=True,
        max_length=TOKENS_MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    ids, masks = inputs["input_ids"], inputs["attention_mask"]

    lightining_model = load_model(Path(CKPTH_DIR, "lightning.pt"))

    torch.set_grad_enabled(False)
    lightining_model.eval()

    with torch.inference_mode():
        logits = lightining_model(ids, masks)

    label = torch.argmax(logits, dim=1)
    prediction = reverse_label_mapper(label)

    return prediction


if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.derterministic = True
    torch.cuda.manual_seed_all(SEED)

    query = {
        "title": "This is the title",
        "keywords": "this;is;the;keywords",
        "content": "This is the content in user query",
    }

    print(inference(query))
