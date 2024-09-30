#!/usr/bin/env python
# coding: utf-8


import os

import comet_ml
import comet_ml.integration.ray

import evaluate
import numpy as np
import ray.train.huggingface.transformers
from datasets import load_dataset
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    enable_full_determinism,
)

comet_ml.login()


# Models
PRE_TRAINED_MODEL_NAME = "google-bert/bert-base-cased"
SEED = 42

enable_full_determinism(SEED)


def get_dataset():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = load_dataset("yelp_review_full")
    dataset["train"] = dataset["train"].shuffle(seed=SEED).select(range(100))
    dataset["test"] = dataset["test"].shuffle(seed=SEED).select(range(100))

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = tokenized_datasets["train"]
    small_eval_dataset = tokenized_datasets["test"]
    return (small_train_dataset, small_eval_dataset)


def train_func(config):
    from comet_ml.integration.ray import comet_worker_logger

    with comet_worker_logger(config):
        small_train_dataset, small_eval_dataset = get_dataset()

        # Model
        model = AutoModelForSequenceClassification.from_pretrained(
            "google-bert/bert-base-cased", num_labels=5
        )

        # Evaluation Metrics
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            experiment = comet_ml.get_running_experiment()
            if experiment:
                experiment.log_confusion_matrix(predictions, labels)

            return metric.compute(predictions=predictions, references=labels)

        # Hugging Face Trainer
        training_args = TrainingArguments(
            do_eval=True,
            do_train=True,
            eval_strategy="epoch",
            num_train_epochs=config["epochs"],
            output_dir="./results",
            overwrite_output_dir=True,
            per_device_eval_batch_size=4,
            per_device_train_batch_size=4,
            report_to=["comet_ml"],
            seed=SEED,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=compute_metrics,
        )

        # Report Metrics and Checkpoints to Ray Train
        callback = ray.train.huggingface.transformers.RayTrainReportCallback()
        trainer.add_callback(callback)

        # Prepare Transformers Trainer
        trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

        # Start Training
        trainer.train()

    comet_ml.get_running_experiment().end()


def train(num_workers: int = 2, use_gpu: bool = False, epochs=1):
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    config = {"use_gpu": use_gpu, "epochs": 2}

    callback = comet_ml.integration.ray.CometTrainLoggerCallback(
        config, project_name="comet-example-ray-train-hugginface-transformers"
    )

    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        train_loop_config=config,
        run_config=RunConfig(callbacks=[callback]),
    )
    return ray_trainer.fit()


ideal_num_workers = 2

available_local_cpu_count = os.cpu_count() - 1
num_workers = min(ideal_num_workers, available_local_cpu_count)

if num_workers < 1:
    num_workers = 1

train(num_workers, use_gpu=False, epochs=5)
