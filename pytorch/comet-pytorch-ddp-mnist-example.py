from comet_ml import Experiment

import os
import argparse

import torch
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn, optim
from collections import OrderedDict
from tqdm import tqdm

torch.manual_seed(0)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


INPUT_SIZE = 784
HIDDEN_SIZES = [128, 64]
OUTPUT_SIZE = 10
BATCH_SIZE = 256


def setup(rank, world_size, backend):
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


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


def train(model, optimizer, criterion, trainloader, epoch):
    model.train()
    for batch_idx, (images, labels) in tqdm(enumerate(trainloader)):
        optimizer.zero_grad()
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        images = images.view(images.size(0), -1)
        pred = model(images)

        loss = criterion(pred, labels)
        loss.backward()

        optimizer.step()


def run(gpu_id, world_size, args):
    """
    This is a single process that is linked to a single GPU

    :param gpu_id: The id of the GPU on the current node
    :param world_size: Total number of processes across nodes
    :param args:
    :return:
    """
    torch.cuda.set_device(gpu_id)

    # The overall rank of this GPU process across multiple nodes
    global_process_rank = args.nr * args.gpus + gpu_id
    if global_process_rank == 0:
        experiment = Experiment(auto_output_logging="simple")

    else:
        experiment = Experiment(disabled=True)

    print(f"Running DDP model on Global Process with Rank: {global_process_rank }.")
    setup(global_process_rank, world_size, args.backend)

    model = build_model()
    model.cuda(gpu_id)
    ddp_model = DDP(model, device_ids=[gpu_id])

    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = optim.Adam(model.parameters())

    # Load training data
    train_dataset = torchvision.datasets.MNIST(
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
        train(ddp_model, optimizer, criterion, trainloader, epoch)

    cleanup()


def get_args():
    parser = argparse.ArgumentParser()
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
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes, starts at 0"
    )
    parser.add_argument(
        "--epochs",
        default=2,
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

    mp.spawn(run, args=(world_size, args), nprocs=args.gpus, join=True)


if __name__ == "__main__":
    main()
