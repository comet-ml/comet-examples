# coding: utf-8
import argparse
import os
import random

import comet_ml
import comet_ml.system.gpu.gpu_logging

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def get_comet_experiment(global_rank):
    """Create Comet Experiment in each worker

    We create the Experiment in the global rank 0 then broadcast the unique
    experiment Key to each worker which creates an ExistingExperiment to logs
    system metrics to the same metrics
    """
    os.environ["COMET_DISTRIBUTED_NODE_IDENTIFIER"] = str(global_rank)
    if global_rank == 0:
        experiment = comet_ml.Experiment(
            project_name="comet-example-pytorch-distributed-torchrun"
        )
        objects = [experiment.get_key()]
    else:
        objects = [None]

    dist.broadcast_object_list(objects, src=0)

    if global_rank != 0:
        experiment = comet_ml.ExistingExperiment(
            experiment_key=objects[0],
            log_env_gpu=True,
            log_env_cpu=True,
            log_env_network=True,
            log_env_details=True,
            log_env_disk=True,
            log_env_host=False,
            display_summary_level=0,
        )

    return experiment


def get_dataset(local_rank, transform):
    # Data should be prefetched
    # Download the data on local rank 0
    if local_rank == 0:
        train_set = torchvision.datasets.CIFAR10(
            root="data", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root="data", train=False, download=False, transform=transform
        )

    # Wait for all local rank to have download the dataset
    dist.barrier()

    # Then load it for other workers on each node
    if local_rank != 0:
        train_set = torchvision.datasets.CIFAR10(
            root="data", train=True, download=False, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root="data", train=False, download=False, transform=transform
        )

    return train_set, test_set


def main():
    num_epochs_default = 5
    batch_size_default = 256  # 1024
    learning_rate_default = 0.1
    random_seed_default = 0

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of training epochs.",
        default=num_epochs_default,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size for one process.",
        default=batch_size_default,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate.",
        default=learning_rate_default,
    )
    parser.add_argument(
        "--random_seed", type=int, help="Random seed.", default=random_seed_default
    )
    parser.add_argument("--cpu", action="store_true", help="Train on CPU only.")
    argv = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    train_on_cpu = argv.cpu

    # We need to use seeds to make sure that the models initialized in
    # different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of sychronizing
    # nodes/GPUs
    if train_on_cpu:
        torch.distributed.init_process_group(backend="gloo")
    else:
        torch.distributed.init_process_group(backend="nccl")

        # We need to set the CUDA device to get distributed communication to works
        torch.cuda.set_device("cuda:{}".format(local_rank))

        # Also inform Comet about which GPU device to keep track of
        comet_ml.system.gpu.gpu_logging.set_devices_to_report([local_rank])

    # Get the Comet experiment for each worker
    comet_experiment = get_comet_experiment(global_rank)

    # Encapsulate the model on the GPU assigned to the current process
    model = torchvision.models.resnet18(pretrained=False)

    if train_on_cpu:
        device = torch.device("cpu")
        model = model.to(device)
        ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        device = torch.device("cuda:{}".format(local_rank))
        model = model.to(device)
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # Prepare dataset and dataloader
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set, test_set = get_dataset(local_rank, transform)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_set)

    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=8
    )
    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(
        dataset=test_set, batch_size=128, shuffle=False, num_workers=8
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5
    )

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))

        # Save and evaluate model routinely
        if local_rank == 0:
            accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
            comet_experiment.log_metric("accuracy", accuracy, epoch=epoch)
            print("-" * 75)
            print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
            print("-" * 75)

        ddp_model.train()

        for step, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if step % 5 == 0:
                print(
                    "Local Rank: {}, Epoch: {}, Step: {}, Loss: {}".format(
                        local_rank, epoch, step, loss
                    )
                )


if __name__ == "__main__":
    main()
