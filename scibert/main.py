import os
import random

import lightning as L
import torch
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.tuner import Tuner
from torchsummary import summary

from scibert.config import *
from scibert.datamodule import ProcodexDataModule
from scibert.features.build_features import build_features
from scibert.model import LightningModel
from scibert.models.dispatcher import MODELS
from scibert.preprocessing.make_data import make
from scibert.utils.logger import logger
from scibert.utils.serializer import pickle_serializer
from scibert.utils.utils import TimingCallback


def training_pipeline(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    traindf, testdf = make(DATA, TEST_DIR)

    train_X, train_y, val_X, val_y, test_X, test_y = build_features(traindf[:-1], testdf[:-1])

    for object, path in zip(
        [(train_X, train_y), (val_X, val_y), (test_X, test_y)],
        [TRAIN_PROCESSED_DATA, VAL_PROCESSED_DATA, TEST_PROCESSED_DATA],
    ):
        pickle_serializer(object=object, path=path, mode="save")

    model = MODELS[MODEL]["model"]

    dm = ProcodexDataModule(batch_size=batch_size)
    lightining_model = LightningModel(model=model, learning_rate=learning_rate)

    # -----------------
    # MODEL TRAINING
    # -----------------

    torch.set_grad_enabled(True)
    lightining_model.train()

    # 1. Save the best model based on maximizing the validation accuracy
    callbacks = [
        ModelCheckpoint(save_top_k=2, mode="max", monitor="val_acc"),
        TimingCallback(),
        # EarlyStopping(monitor="val_acc", mode="max"),
    ]

    # Overfitting to minibatch is a sanity check that assumes:
    # A good model should always be able to fit one minibatch with very high train accuracy, If the model cant fit
    trainer = L.Trainer(
        # overfit_batches=1,
        max_epochs=epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=5,
        deterministic=True,
        logger=MLFlowLogger(),
    )

    # Using tuner to determine the appropriate learning rates
    tuner = Tuner(trainer)
    lightining_model.learning_rate = tuner.lr_find(lightining_model, datamodule=dm).suggestion()

    trainer.fit(model=lightining_model, datamodule=dm)
    logger.info("Training complete!")

    # trainer.test(model=lightining_model, datamodule=dm, ckpt_path="best")

    return trainer.checkpoint_callback.best_model_path


def evaluation_pipeline(best_model):
    model = MODELS[MODEL]["model"]

    lightining_model = LightningModel.load_from_checkpoint(best_model, model=model)

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
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.derterministic = True
    torch.cuda.manual_seed_all(SEED)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    best_model = training_pipeline(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

    evaluation_pipeline(best_model)


if __name__ == "__main__":
    main()
