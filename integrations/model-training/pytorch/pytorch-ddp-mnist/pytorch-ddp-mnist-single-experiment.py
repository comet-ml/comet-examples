# -*- coding: utf-8 -*-
import argparse
import hashlib
import os
from collections import OrderedDict

import comet_ml

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from tqdm import tqdm

torch.manual_seed(0)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

PROJECT_NAME = "comet-example-ddp-mnist-single"
INPUT_SIZE = 784
HIDDEN_SIZES = [128, 64]
OUTPUT_SIZE = 10
BATCH_SIZE = 256


def get_experiment(run_id):
    experiment_id = hashlib.sha1(run_id.encode("utf-8")).hexdigest()
    os.environ["COMET_EXPERIMENT_KEY"] = experiment_id

    api = comet_ml.API()  # Assumes API key is set in config/env
    api_experiment = api.get_experiment_by_id(experiment_id)

    if api_experiment is None:
        return comet_ml.Experiment(project_name=PROJECT_NAME)

    else:
        return comet_ml.ExistingExperiment(project_name=PROJECT_NAME)


def setup():
    # initialize the process group
    dist.init_process_group(backend="nccl", init_method="env://")


def cleanup():
    dist.destroy_process_group()


def build_model():
    model = nn.Sequential(
        OrderedDict(
            [
                ("linear0", nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0])),
                ("activ0", nn.ReLU()),
                ("linear1", nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1])),
                ("activ1", nn.ReLU()),
                ("linear2", nn.Linear(HIDDEN_SIZES[1], OUTPUT_SIZE)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

    return model


def train(model, optimizer, criterion, trainloader, epoch, process_rank, experiment):
    model.train()
    for batch_idx, (images, labels) in tqdm(enumerate(trainloader)):
        optimizer.zero_grad()
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        images = images.view(images.size(0), -1)
        pred = model(images)

        loss = criterion(pred, labels)
        loss.backward()

        experiment.log_metric(f"{process_rank}_train_batch_loss", loss.item())

        optimizer.step()


def run(local_rank, args):
    """
    This is a single process that is linked to a single GPU

    :param local_rank: The id of the GPU on the current node
    :param args:
    :return:
    """
    setup()

    # The overall rank of this GPU process across multiple nodes
    world_size = dist.get_world_size()
    global_process_rank = dist.get_rank()
    print(f"Running DDP model on Global Process with Rank: {global_process_rank }.")

    # Set GPU to local device
    torch.cuda.set_device(f"cuda:{local_rank}")

    experiment = args.experiment

    model = build_model()
    model.cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = optim.Adam(ddp_model.parameters())

    # Load training data
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=global_process_rank
    )
    trainloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )

    for epoch in range(1, args.epochs + 1):
        train(
            ddp_model,
            optimizer,
            criterion,
            trainloader,
            epoch,
            global_process_rank,
            experiment,
        )

    cleanup()
    experiment.end()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str)
    parser.add_argument("-b", "--backend", type=str, default="nccl")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="8892",
        help="""Port that master is listening on, will default to 29500 if not
           provided. Master must be able to accept network traffic on the
           host and port.""",
    )
    parser.add_argument(
        "-n",
        "--nodes",
        default=1,
        type=int,
        metavar="N",
        help="total number of compute nodes",
    )
    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "-nr",
        "--node_rank",
        default=0,
        type=int,
        help="ranking within the nodes, starts at 0",
    )
    parser.add_argument(
        "-lr",
        "--local_rank",
        default=0,
        type=int,
        help="rank of the process within the current node, starts at 0",
    )
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    return parser.parse_args()


def main():
    args = get_args()

    world_size = args.gpus * args.nodes
    global_process_rank = args.node_rank * args.gpus + args.local_rank

    experiment = get_experiment(args.run_id)

    # Make sure all nodes can talk to each other on the unprivileged port range
    # (1024-65535) in addition to the master port
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(global_process_rank)

    args.experiment = experiment
    run(args.local_rank, args)


if __name__ == "__main__":
    main()
