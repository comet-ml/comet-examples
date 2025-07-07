# coding: utf-8
"""Pytorch Distributed Data Parallel Example with Learning Rate Scaling

"""
import argparse
import os
import dotenv

dotenv.load_dotenv()

import comet_ml
from comet_ml import Experiment

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

torch.manual_seed(0)

# This is the batch size being used per GPU
LEARNING_RATE = 0.001

# Learning Rate scaling factor is computed relative to this batch size
MIN_BATCH_SIZE = 8

comet_ml.login(api_key=os.getenv("COMET_API_KEY_TEST"), workspace="fschlz")


def get_device(local_rank):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    # DDP on MPS is not supported, so fall back to CPU
    if dist.is_initialized():
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scale_lr(batch_size, lr):
    return lr * (batch_size / MIN_BATCH_SIZE)


def setup(rank, world_size, backend):
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def load_data(data_dir="./data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    return trainset, testset


def train(model, optimizer, criterion, trainloader, epoch, device, experiment):
    model.train()
    total_loss = 0
    epoch_steps = 0
    for batch_idx, (images, labels) in tqdm(enumerate(trainloader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)

        loss = criterion(pred, labels)
        loss.backward()

        experiment.log_metric("train_batch_loss", loss.item(), step=epoch_steps)
        total_loss += loss.item()
        epoch_steps += 1

    return total_loss / len(trainloader)


def test(model, criterion, testloader, epoch, device, experiment):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)
            loss = criterion(pred, labels)
            total_loss += loss.item()

    return total_loss / len(testloader)


def run(local_rank, world_size, args):
    print(f"Running DDP example on rank {local_rank}.")

    backend = args.backend
    if not torch.cuda.is_available():
        backend = "gloo"

    setup(local_rank, world_size, backend)

    device = get_device(local_rank)

    # Get Data
    trainset, testset = load_data()

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=local_rank
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.replica_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        testset, num_replicas=world_size, rank=local_rank
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.replica_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=test_sampler,
    )

    # build the model
    model = Net().to(device)

    # wrap the model in DDP
    # device_ids tell DDP where to move the data for the replica
    if torch.cuda.is_available():
        ddp_model = DDP(model, device_ids=[local_rank])
    else:
        ddp_model = DDP(model)

    # build the optimizer
    scaled_lr = scale_lr(args.replica_batch_size * world_size, LEARNING_RATE)
    optimizer = optim.SGD(ddp_model.parameters(), lr=scaled_lr, momentum=0.9)

    # loss function
    criterion = nn.CrossEntropyLoss()

    experiment = None
    if local_rank == 0:
        experiment = comet_ml.Experiment()
        experiment.set_name(f"pytorch-ddp-cifar10-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        experiment.log_parameters(
            {
                "world_size": world_size,
                "epochs": args.epochs,
                "replica_batch_size": args.replica_batch_size,
                "node_rank": args.node_rank,
                "gpus_per_node": args.gpus,
                "learning_rate": LEARNING_RATE,
                "scaled_learning_rate": scaled_lr,
            }
        )

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train(
            ddp_model, optimizer, criterion, trainloader, epoch, device, experiment
        )

        if local_rank == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss}")
            experiment.log_metric("train_loss", train_loss, epoch=epoch)

            test_loss = test(ddp_model, criterion, testloader, epoch, device, experiment)
            print(f"Epoch {epoch}, Test Loss: {test_loss}")
            experiment.log_metric("test_loss", test_loss, epoch=epoch)

    cleanup()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str)
    parser.add_argument("-b", "--backend", type=str, default="gloo")
    parser.add_argument(
        "-n",
        "--nodes",
        default=1,
        type=int,
        metavar="N",
        help="total number of compute nodes",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        default=1,
        type=int,
        help="number of gpus per node (or processes to use for CPU)",
    )
    parser.add_argument(
        "-nr",
        "--node_rank",
        default=0,
        type=int,
        help="ranking within the nodes, starts at 0",
    )
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--replica_batch_size",
        default=32,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
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
        provided. Master must be able to accept network traffic on the host and
        port.""",
    )
    return parser.parse_args()


def main():
    args = get_args()
    world_size = args.gpus * args.nodes

    # Make sure all nodes can talk to each other on the unprivileged port range
    # (1024-65535) in addition to the master port
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    mp.spawn(
        run,
        args=(
            world_size,
            args,
        ),
        nprocs=args.gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
