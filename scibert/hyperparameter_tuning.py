import os
import random

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from scibert.config import *
from scibert.datamodule import ProcodexDataModule
from scibert.model import LightningModel
from scibert.models.dispatcher import MODELS


def initialize(seed: int) -> str:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.derterministic = True
    torch.cuda.manual_seed_all(seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    initialize(SEED)

    cli = LightningCLI(
        model_class=LightningModel,
        datamodule_class=ProcodexDataModule,
        run=False,
        save_config_callback=None,
        seed_everything_default=SEED,
        trainer_defaults={
            "max_epochs": 10,
            "accelerator": "auto",
            "callbacks": [ModelCheckpoint(save_top_k=5, mode="max", monitor="val_acc")],
        },
    )

    model = MODELS[MODEL]["model"]
    lightning_model = LightningModel(model=model, learning_rate=cli.model.learning_rate)

    cli.trainer.fit(lightning_model, datamodule=cli.datamodule)
    cli.trainer.test(
        lightning_model,
        datamodule=cli.datamodule,
    )

    # python3 -m scibert.tuning --model.learning_rate 0.1
