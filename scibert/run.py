import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import os
from scibert.preprocessing.make_data import make
from scibert.utils.logger import logger
from scibert.utils.serializer import pickle_serializer

from scibert.config import *
from scibert.features.build_features import build_features
from scibert.models.dispatcher import MODELS

import lightning as L
import torchmetrics
import torch.nn.functional as F

from lightning.pytorch.loggers import MLFlowLogger


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

    def prepare_data(self):
        if os.path.exists(PROCESSED_DATA):
            (
                trainIds,
                trainMask,
                trainLabel,
                testIds,
                testMask,
                testLabel,
            ) = pickle_serializer(
                path=PROCESSED_DATA,
                mode="load",
            )
        else:
            traindf, testdf = make(DATA, TEST_DIR)

            train_X, train_y, test_X, test_y = build_features(
                traindf[:150], testdf[:150]
            )

            trainIds, trainMask, trainLabel = self.generate_inputs(train_X, train_y)
            testIds, testMask, testLabel = self.generate_inputs(test_X, test_y)

            pickle_serializer(
                object=(
                    trainIds,
                    trainMask,
                    trainLabel,
                    testIds,
                    testMask,
                    testLabel,
                ),
                path="data/processed/generated_inputs.pkl",
                mode="save",
            )

        self.traininputs = {
            "ids": trainIds,
            "masks": trainMask,
            "label": trainLabel,
        }
        self.testinputs = {"ids": testIds, "masks": testMask, "label": testLabel}

    def generate_inputs(self, X: np.ndarray, y: np.ndarray) -> list:
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

            # target = torch.zeros(2)
            # target[y[i]] = 1
            targets.append(y[i])

        input_ids = torch.stack(input_ids, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        # targets = torch.stack(targets, dim=0)
        targets = torch.from_numpy(np.array(targets))
        return input_ids, attention_masks, targets

    def setup(self, stage: str):
        self.traindataset = TensorDataset(
            self.traininputs["ids"],
            self.traininputs["masks"],
            self.traininputs["label"],
        )
        self.testdataset = TensorDataset(
            self.testinputs["ids"],
            self.testinputs["masks"],
            self.testinputs["label"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.traindataset,
            batch_size=BATCH_SIZE,
            num_workers=10,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.testdataset,
            batch_size=BATCH_SIZE,
            num_workers=10,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testdataset,
            batch_size=BATCH_SIZE,
            num_workers=10,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.testdataset,
            batch_size=BATCH_SIZE,
            num_workers=10,
        )


def compute_accuracy(model, dataloader, device=None):
    if device is None:
        device = torch.device("cpu")

    model = model.eval()

    acc = torchmetrics.Accuracy(task="binary", num_classes=2)

    for idx, (ids, masks, labels) in enumerate(dataloader):
        ids, masks, labels = ids.to(device), masks.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(ids, masks)

        predictions = torch.argmax(logits, dim=1)

        acc(predictions, labels)

    return acc


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="binary", num_classes=2)

    def forward(self, ids, masks):
        return self.model(ids, masks)

    def _shared_step(self, batch):
        ids, masks, true_labels = batch
        logits = self.forward(ids, masks)
        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return (loss, true_labels, predicted_labels)

    def training_step(self, batch, batch_idx):  # receives a mini batch of data
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss, prog_bar=True)

        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)

        self.test_acc(predicted_labels, true_labels)
        self.log("val_acc", self.test_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("test_loss", loss, prog_bar=True)

        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


def training_pipeline() -> None:
    dm = ProcodexDataModule(batch_size=32)

    model = MODELS[MODEL]["model"]

    lightining_model = LightningModel(model=model, learning_rate=LEARNING_RATE)

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=5,
        deterministic=True,
        logger=MLFlowLogger(),
    )

    trainer.fit(model=lightining_model, datamodule=dm)

    train_acc = trainer.validate(dataloaders=dm.train_dataloader(), ckpt_path="best")[
        0
    ]["val_acc"]
    val_acc = trainer.validate(datamodule=dm, ckpt_path="best")[0]["val_acc"]
    test_acc = trainer.test(datamodule=dm, ckpt_path="best")[0]["test_acc"]
    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )

    torch.save(model.state_dict(), Path(CKPTH_DIR, "lightning.pt"))
    logger.info("Training complete!")


def main():
    initialize(SEED)

    training_pipeline()


if __name__ == "__main__":
    main()
