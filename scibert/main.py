import os
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from scibert.config import *
from scibert.features.build_features import build_features
from scibert.models.dispatcher import MODELS
from scibert.preprocessing.make_data import make
from scibert.utils.logger import logger
from scibert.utils.serializer import pickle_serializer
from scibert.utils.utils import TimingCallback


def initialize(seed: int) -> str:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.derterministic = True
    torch.cuda.manual_seed_all(seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ProcodexDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()

        self.batch_size = batch_size

    def generate_model_inputs(self, X: np.ndarray, y: np.ndarray) -> list:
        input_ids = []
        attention_masks = []
        targets = []

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


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")

    def forward(self, ids, masks):
        return self.model(ids, masks)

    def _shared_step(self, batch):
        ids, masks, true_labels = batch
        logits = self.forward(ids, masks)
        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return (loss, true_labels, predicted_labels)

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.train_acc(predicted_labels, true_labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.val_acc(predicted_labels, true_labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


def training_pipeline() -> None:
    traindf, testdf = make(DATA, TEST_DIR)

    train_X, train_y, val_X, val_y, test_X, test_y = build_features(traindf[:100], testdf[:100])

    for object, path in zip(
        [(train_X, train_y), (val_X, val_y), (test_X, test_y)],
        [TRAIN_PROCESSED_DATA, VAL_PROCESSED_DATA, TEST_PROCESSED_DATA],
    ):
        pickle_serializer(object=object, path=path, mode="save")

    dm = ProcodexDataModule(batch_size=32)

    model = MODELS[MODEL]["model"]

    lightining_model = LightningModel(model=model, learning_rate=LEARNING_RATE)

    checkpoint_callbacks = ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc")

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callbacks, TimingCallback()],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=5,
        deterministic=True,
        logger=MLFlowLogger(),
    )

    trainer.fit(model=lightining_model, datamodule=dm)

    trainer.save_checkpoint(Path(CKPTH_DIR, "lightning.pt"))

    logger.info("Training complete!")


def evaluation_pipeline():
    model = MODELS[MODEL]["model"]

    lightining_model = LightningModel.load_from_checkpoint(Path(CKPTH_DIR, "lightning.pt"), model=model)

    dm = ProcodexDataModule()
    dm.setup("test")

    test_loader = dm.test_dataloader()
    acc = torchmetrics.Accuracy(task="binary", num_classes=2)
    cm = torchmetrics.ConfusionMatrix(task="binary")

    lightining_model.eval()

    for batch in test_loader:
        ids, masks, true_labels = batch

        with torch.inference_mode():
            logits = lightining_model.forward(ids, masks)

        predicted_labels = torch.argmax(logits, dim=1)

        acc(predicted_labels, true_labels)
        cm(predicted_labels, true_labels)

    print(predicted_labels, acc.compute(), cm.compute())


def main():
    initialize(SEED)

    training_pipeline()

    evaluation_pipeline()


if __name__ == "__main__":
    main()
