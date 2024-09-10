# coding: utf-8

import os

import comet_ml

from torch import nn, optim
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.data import ToTensor
from catalyst.loggers.comet import CometLogger

comet_ml.login()
logger = CometLogger(logging_frequency=10)

model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)


loaders = {
    "train": DataLoader(
        MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()),
        batch_size=32,
    ),
    "valid": DataLoader(
        MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
        batch_size=32,
    ),
}

runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets", loss_key="loss"
)

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=1,
    hparams={
        "lr": 0.02,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0,
        "amsgrad": False,
    },
    callbacks=[
        dl.AccuracyCallback(
            input_key="logits", target_key="targets", topk_args=(1, 3, 5)
        ),
        dl.PrecisionRecallF1SupportCallback(
            input_key="logits", target_key="targets", num_classes=10
        ),
    ],
    logdir="./logs",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
    load_best_on_end=True,
    loggers={"comet": logger},
)
