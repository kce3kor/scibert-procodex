import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics

from scibert.config import *
from scibert.models.dispatcher import MODELS


class LightningModel(L.LightningModule):
    def __init__(self, model=None, hidden_units=None, learning_rate=None):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")

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
        self.log("train_loss", loss, prog_bar=True)

        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)

        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("test_loss", loss, prog_bar=True)

        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        ## SGD
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)

        ## Adam
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # TYPES OF SCHEDULERS: StepLR, Reduce-on-plateau , Cosine Annealing
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        ## step_size = epochs, so we are reducing by 0.5 in every 10 epochs

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, mode="max")
        ## If there is no improvement in 5 epochs, reduce it by 10%, track the improvement using val_acc with "max", for train_acc == "min"

        # num_steps = (EPOCHS * len(dm.train_dataloader()),)
        # load num_steps from outside the class
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        ## Reduce the lr in a cosine format
        ## set the interval as STEP

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss", "interval": "step", "frequency": 1},
        }
