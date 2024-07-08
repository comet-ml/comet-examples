# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.ml
#  Copyright (C) 2015-2020 Comet ML INC
#  This file can not be copied and/or distributed without
#  the express permission of Comet ML Inc.
# *******************************************************

import os

import comet_ml
from comet_ml import Optimizer

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_lightning import LightningModule, Trainer

# Login to Comet if needed
comet_ml.login(project_name="comet-example-pytorch-lightning-optimizer")


class PyTorchLightningModel(LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {"loss": F.cross_entropy(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(
            MNIST(
                os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
            ),
            batch_size=32,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


optimizer_config = {
    "algorithm": "bayes",
    "spec": {"maxCombo": 5},
    "parameters": {
        "learning_rate": {"min": 0.01, "max": 0.99, "type": "double", "gridSize": 10}
    },
}


def run():
    optimizer = Optimizer(optimizer_config)

    for parameters in optimizer.get_parameters():
        model = PyTorchLightningModel(**parameters["parameters"])

        comet_logger = CometLogger(
            optimizer_data=parameters,
        )

        trainer = Trainer(
            max_epochs=1,
            logger=[comet_logger],
        )

        trainer.fit(model)


if __name__ == "__main__":
    run()
