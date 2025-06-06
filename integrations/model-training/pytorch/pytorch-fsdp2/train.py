# coding: utf-8
import argparse
import os
import time

# Import comet_ml at the top of the file
import comet_ml

import torch
from checkpoint import Checkpointer
from model import ModelArgs, Transformer
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from utils import inspect_mixed_precision, inspect_model


def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def setup_comet_logging(config):
    global_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Initialize experiment key
    experiment_key = [None]

    if global_rank == 0:
        # Create experiment configuration
        experiment_config = comet_ml.ExperimentConfig(
            distributed_node_identifier=global_rank,
        )

        # Create the experiment on first worker
        experiment = comet_ml.start(
            project_name=config["comet_project"], experiment_config=experiment_config
        )

        # Get experiment key to share with other workers
        experiment_key[0] = experiment.get_key()
        print(f"Rank {global_rank}: Created experiment with key: {experiment_key[0]}")

        # Log system info and environment details
        experiment.log_other("world_size", world_size)
        experiment.log_other(
            "cuda_version", torch.version.cuda if torch.cuda.is_available() else "N/A"
        )
        experiment.log_other("pytorch_version", torch.__version__)

        # Log hyperparameters from the config dictionary
        params = {
            "vocab_size": config["vocab_size"],
            "batch_size": config["batch_size"],
            "seq_len": config["seq_len"],
            "n_layers": config["n_layers"],
            "n_heads": config["n_heads"],
            "max_seq_len": config["seq_len"],
            "dropout_p": config["dropout_p"],
            "mixed_precision": config["mixed_precision"],
            "explicit_prefetching": config["explicit_prefetching"],
            "dcp_api": config["dcp_api"],
        }
        experiment.log_parameters(params)

    # Broadcast the experiment key to all workers
    torch.distributed.broadcast_object_list(experiment_key, src=0)

    # Create experiment with same key on non-zero ranks
    if global_rank != 0:
        print(f"Rank {global_rank}: Received experiment key: {experiment_key[0]}")

        experiment_config_kwargs = {
            "log_env_gpu": True,
            "log_env_cpu": True,
            "log_env_network": True,
            "log_env_disk": True,
            "log_env_host": False,
            "log_env_details": True,
            "display_summary_level": 0,
            "distributed_node_identifier": global_rank,
        }
        experiment = comet_ml.start(
            experiment_key=experiment_key[0],
            experiment_config=comet_ml.ExperimentConfig(**experiment_config_kwargs),
        )
        print(f"Rank {global_rank}: Connected to the same experiment")

    # Synchronize to ensure all processes have created their experiment
    torch.distributed.barrier()
    return experiment


def main(args):
    # Convert args namespace to a dictionary for easier handling
    config = vars(args)

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    # Get global rank for proper distributed training coordination
    global_rank = torch.distributed.get_rank()
    torch.manual_seed(0)

    # Add model hyperparameters to the config dictionary
    config.update(
        {
            # Model hyperparameters
            "vocab_size": 1024,
            "batch_size": 32,
            "seq_len": 64,
            "n_layers": 10,
            "n_heads": 4,
            "dropout_p": 0,
        }
    )

    # Create model args from config hyperparameters
    model_args = ModelArgs(
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        vocab_size=config["vocab_size"],
        max_seq_len=config["seq_len"],
        dropout_p=config["dropout_p"],
    )

    experiment = setup_comet_logging(config)

    with torch.device("meta"):
        model = Transformer(model_args)

    fsdp_kwargs = {}
    if config["mixed_precision"]:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    inspect_model(model)

    # Log model architecture summary with Comet if rank 0
    if global_rank == 0 and experiment is not None:
        # Log model summary as a string
        model_summary = str(model)
        experiment.set_model_graph(model_summary)
        experiment.log_other("model_layers", model_args.n_layers)
        experiment.log_other("model_heads", model_args.n_heads)

    if config["explicit_prefetching"]:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

    checkpointer = Checkpointer("checkpoints", dcp_api=config["dcp_api"])
    if checkpointer.last_training_time is None:
        model.to_empty(device="cuda")
        model.reset_parameters()
    else:
        checkpointer.load_model(model)

    if config["mixed_precision"]:
        inspect_mixed_precision(model)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    if checkpointer.last_training_time is not None:
        checkpointer.load_optim(model, optim)

    for step in range(100):
        if config["explicit_prefetching"]:
            model.unshard()
        x = torch.randint(
            0,
            config["vocab_size"],
            (config["batch_size"], config["seq_len"]),
            device=device,
        )
        loss = model(x).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()

        # Log metrics only on the first worker (global rank 0)
        if global_rank == 0 and experiment is not None:
            experiment.log_metric("loss", loss.item(), step=step)
            # Log learning rate
            experiment.log_metric(
                "learning_rate", optim.param_groups[0]["lr"], step=step
            )

        time.sleep(1)

    checkpointer.save(model, optim)

    # End the Comet experiment if it exists
    if global_rank == 0 and experiment is not None:
        experiment.end()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    parser.add_argument(
        "--comet-project",
        type=str,
        default="comet-example-pytorch-fsdp2",
        help="Comet project name",
    )
    args = parser.parse_args()
    main(args)
