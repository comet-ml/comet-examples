# -*- coding: utf-8 -*-
from comet_ml import Optimizer


def run():

    opt_config = {
        # We pick the Bayes algorithm:
        "algorithm": "grid",
        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "x": {"type": "integer", "min": 1, "max": 5},
        },
        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "loss",
            "objective": "minimize",
        },
    }

    # initialize the optimizer object
    opt = Optimizer(config=opt_config)

    # print Optimizer id
    optimizer_id = opt.get_id()
    print(optimizer_id)


if __name__ == "__main__":
    run()
