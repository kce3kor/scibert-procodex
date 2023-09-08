import os

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from scibert.config import *
from scibert.models.dispatcher import MODELS
from scibert.utils.serializer import pickle_serializer


class ProcodexDataModule(L.LightningDataModule):
    def __init__(self, batch_size=8):
        super().__init__()

        self.batch_size = batch_size

    def generate_model_inputs(self, X: np.ndarray, y: np.ndarray) -> list:
        input_ids, attention_masks, targets = [], [], []

        tokenizer = MODELS[MODEL]["tokenizer"]

        for i in tqdm(range(len(X))):
            X[i] = X[i].strip()

            inputs = tokenizer.encode_plus(
                X[i],
                add_special_tokens=True,
                max_length=TOKENS_MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids.append(inputs["input_ids"].squeeze(0))
            attention_masks.append(inputs["attention_mask"].squeeze(0))
            targets.append(y[i])

        input_ids = torch.stack(input_ids, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        targets = torch.from_numpy(np.array(targets))

        return input_ids, attention_masks, targets

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit" or stage == None:
            if os.path.exists(TRAIN_PROCESSED_DATA) or os.path.exists(VAL_PROCESSED_DATA):
                train_X, train_y = pickle_serializer(
                    path=TRAIN_PROCESSED_DATA,
                    mode="load",
                )
                val_X, val_y = pickle_serializer(
                    path=VAL_PROCESSED_DATA,
                    mode="load",
                )
            else:
                raise Exception(f"File not found at {TRAIN_PROCESSED_DATA}, {VAL_PROCESSED_DATA}")

            trainIds, trainMask, trainLabel = self.generate_model_inputs(train_X, train_y)
            valIds, valMask, valLabel = self.generate_model_inputs(val_X, val_y)

            self.traindataset = TensorDataset(
                trainIds,
                trainMask,
                trainLabel,
            )
            self.valdataset = TensorDataset(valIds, valMask, valLabel)

        if stage == "test" or stage == None:
            if os.path.exists(TEST_PROCESSED_DATA):
                test_X, test_y = pickle_serializer(
                    path=TEST_PROCESSED_DATA,
                    mode="load",
                )
            else:
                raise Exception(f"File not found at {TEST_PROCESSED_DATA}")

            testIds, testMask, testLabel = self.generate_model_inputs(test_X, test_y)

            self.testdataset = TensorDataset(
                testIds,
                testMask,
                testLabel,
            )

    def train_dataloader(self):
        return DataLoader(
            self.traindataset,
            batch_size=self.batch_size,
            num_workers=10,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valdataset,
            batch_size=self.batch_size,
            num_workers=10,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testdataset,
            batch_size=self.batch_size,
            num_workers=10,
        )

    def predict_dataloader(self):
        pass
