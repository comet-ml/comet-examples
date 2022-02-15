"""Pytorch Distributed Data Parallel Example with Learning Rate Scaling

"""
from comet_ml import Experiment

import os
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn, optim
from tqdm import tqdm

torch.manual_seed(0)

# This is the batch size being used per GPU
LEARNING_RATE = 0.001

# Learning Rate scaling factor is computed relative to this batch size
MIN_BATCH_SIZE = 8


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


def train(model, optimizer, criterion, trainloader, epoch, gpu_id, experiment):
    model.train()
    total_loss = 0
    epoch_steps = 0
    for batch_idx, (images, labels) in tqdm(enumerate(trainloader)):
        optimizer.zero_grad()
        images = images.cuda(gpu_id, non_blocking=True)
        labels = labels.cuda(gpu_id, non_blocking=True)

        pred = model(images)

        loss = criterion(pred, labels)
        loss.backward()

        experiment.log_metric("train_batch_loss", loss.item())

        total_loss += loss.item()
        epoch_steps += 1

        optimizer.step()

    return total_loss / epoch_steps


def evaluate(model, criterion, valloader, epoch, local_rank):
    # Validation loss
    total_loss = 0.0
    epoch_steps = 0
    total = 0
    correct = 0

    model.eval()
    for i, data in enumerate(valloader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.cuda(local_rank), labels.cuda(local_rank)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            epoch_steps += 1

    val_acc = correct / total
    val_loss = total_loss / epoch_steps

    return val_loss, val_acc


def test_accuracy(net, testset, device="cpu"):
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def run(local_rank, world_size, args):
    """
    This is a single process that is linked to a single GPU

    :param local_rank: The id of the GPU on the current node
    :param world_size: Total number of processes across nodes
    :param args:
    :return:
    """
    torch.cuda.set_device(local_rank)

    # The overall rank of this GPU process across multiple nodes
    global_process_rank = args.node_rank * args.gpus + local_rank

    experiment = Experiment(auto_output_logging="simple")
    experiment.log_parameter("run_id", args.run_id)
    experiment.log_parameter("global_process_rank", global_process_rank)
    experiment.log_parameter("replica_batch_size", args.replica_batch_size)
    experiment.log_parameter("batch_size", args.replica_batch_size * world_size)

    learning_rate = scale_lr(args.replica_batch_size * world_size, LEARNING_RATE)
    experiment.log_parameter("learning_rate", learning_rate)

    print(f"Running DDP model on Global Process with Rank: {global_process_rank }.")
    setup(global_process_rank, world_size, args.backend)

    model = Net()
    model.cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9)

    # Load training data
    trainset, testset = load_data()
    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_subset, num_replicas=world_size, rank=global_process_rank
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.replica_batch_size,
        sampler=train_sampler,
        num_workers=8,
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=args.replica_batch_size, shuffle=True, num_workers=8
    )

    for epoch in range(args.epochs):
        train_loss = train(
            ddp_model, optimizer, criterion, trainloader, epoch, local_rank, experiment
        )
        experiment.log_metric("train_loss", train_loss)

        val_loss, val_acc = evaluate(ddp_model, criterion, valloader, epoch, local_rank)
        experiment.log_metric("val_loss", val_loss, epoch=epoch)
        experiment.log_metric("val_acc", val_acc, epoch=epoch)

    test_acc = test_accuracy(model, testset, f"cuda:{local_rank}")
    experiment.log_metric("test_acc", test_acc, epoch=args.epochs)

    cleanup()


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

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
    parser.add_argument("-b", "--backend", type=str, default="nccl")
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
        provided. Master must be able to accept network traffic on the host and port.""",
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
        run, args=(world_size, args,), nprocs=args.gpus, join=True,
    )


if __name__ == "__main__":
    main()
