#!/usr/bin/env python
# coding: utf-8

import os

import comet_ml

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch.loggers import CometLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

comet_ml.login(project_name="comet-example-pytorch-lightning")


# Arguments made to CometLogger are passed on to the comet_ml.Experiment class
comet_logger = CometLogger()


class Model(pl.LightningModule):
    def __init__(self, layer_size=784):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = torch.nn.Linear(layer_size, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.device_count() else 64


# Init our model
model = Model()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(
    PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

eval_ds = MNIST(
    PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor()
)
eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

comet_logger.log_hyperparams({"batch_size": BATCH_SIZE})

# Initialize a trainer
trainer = Trainer(max_epochs=3, logger=comet_logger)

# Train the model ⚡
trainer.fit(model, train_loader, eval_loader)
