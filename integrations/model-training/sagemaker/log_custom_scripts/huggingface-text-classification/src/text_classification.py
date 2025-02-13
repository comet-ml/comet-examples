# coding: utf-8

import argparse
import json
import logging
import os
import sys

import comet_ml

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

if __name__ == "__main__":

    SM_TRAINING_ENV = json.loads(os.getenv("SM_TRAINING_ENV"))
    SM_TRAINING_JOB_NAME = SM_TRAINING_ENV.get("job_name")
    SM_BUCKET = os.getenv("SM_BUCKET", "")

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments
    # to the script
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument(
        "--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)

        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def log_metrics(results):
        experiment = comet_ml.get_running_experiment()
        experiment.log_metrics(results)

    def log_sagemaker_metadata():
        experiment = comet_ml.get_running_experiment()
        experiment.log_others(SM_TRAINING_ENV)

    def _get_model_metadata():
        model_uri = f"s3://{SM_BUCKET}/{SM_TRAINING_JOB_NAME}/output/model.tar.gz"
        return {"model_uri": model_uri}

    def log_model(name, model):
        experiment = comet_ml.get_running_experiment()
        comet_ml.integration.pytorch.log_model(
            experiment, model, name, metadata=_get_model_metadata()
        )

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        report_to="comet_ml",
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset, prefix="eval")
    log_metrics(eval_result)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print("***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.model_dir)
    log_model("imdb-classifier", trainer.model, {})
