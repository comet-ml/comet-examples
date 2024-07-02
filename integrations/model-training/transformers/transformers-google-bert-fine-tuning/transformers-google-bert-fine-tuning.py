# coding: utf-8
import comet_ml

import evaluate
import numpy as np
from datasets import load_dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    enable_full_determinism,
)

SEED = 42

enable_full_determinism(SEED)

# Login to Comet if needed
comet_ml.init(project_name="comet-example-transformers-google-bert-fine-tuning")


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


dataset = load_dataset("yelp_review_full")
dataset["train"] = dataset["train"].shuffle(seed=SEED).select(range(100))
dataset["test"] = dataset["test"].shuffle(seed=SEED).select(range(100))

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"]
small_eval_dataset = tokenized_datasets["test"]

model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", num_labels=5
)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    experiment = comet_ml.get_running_experiment()
    if experiment:
        experiment.log_confusion_matrix(predictions, labels)

    return metric.compute(predictions=predictions, references=labels)


EPOCHS = 3

training_args = TrainingArguments(
    seed=SEED,
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    do_train=True,
    do_eval=True,
    report_to=["all"],
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
