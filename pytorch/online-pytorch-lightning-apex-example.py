# -*- coding: utf-8 -*-
# Copyright (C) 2018-2020 Nvidia
# Released under BSD-3 license https://github.com/NVIDIA/apex/blob/master/LICENSE

from comet_ml import Experiment

import torch
from apex import amp


def run():
    experiment = Experiment()

    torch.cuda.set_device("cuda:0")

    torch.backends.cudnn.benchmark = True

    N, D_in, D_out = 64, 1024, 16

    # Each process receives its own batch of "fake input data" and "fake target data."
    # The "training loop" in each process just uses this fake batch over and over.
    # https://github.com/NVIDIA/apex/tree/master/examples/imagenet provides a more
    # realistic example of distributed data sampling for both training and validation.
    x = torch.randn(N, D_in, device="cuda")
    y = torch.randn(N, D_out, device="cuda")

    model = torch.nn.Linear(D_in, D_out).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    loss_fn = torch.nn.MSELoss()

    for t in range(5000):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

    print("final loss = ", loss)
    experiment.log_metric("final_loss", loss)


if __name__ == "__main__":
    run()
