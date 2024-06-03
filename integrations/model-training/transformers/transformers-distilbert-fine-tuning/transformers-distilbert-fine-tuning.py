# coding: utf-8
import warnings

import comet_ml

import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

EPOCHS = 100

# Login to Comet if needed
comet_ml.init(project_name="comet-example-transformers-distilbert-fine-tuning")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocess(texts, labels):
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return encoded, torch.tensor(labels)


def compute_metrics(pred):
    experiment = comet_ml.get_global_experiment()

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)

    if experiment:
        experiment.log_confusion_matrix(preds, labels)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


warnings.filterwarnings("ignore")

PRE_TRAINED_MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

df = pd.read_csv("data/title_conference.csv")
df.head()
df["Conference"] = pd.Categorical(df["Conference"])
df["Target"] = df["Conference"].cat.codes

train_data, test_data = train_test_split(df, test_size=0.01, stratify=df["Target"])
train_texts, train_labels = (
    train_data["Title"].values.tolist(),
    train_data["Target"].values.tolist(),
)
test_texts, test_labels = (
    test_data["Title"].values.tolist(),
    test_data["Target"].values.tolist(),
)
train_encoded, train_labels = preprocess(train_texts, train_labels)
test_encoded, test_labels = preprocess(test_texts, test_labels)
train_dataset = Dataset(train_encoded, train_labels)
test_dataset = Dataset(test_encoded, test_labels)

indices = torch.arange(10)
train_dataset = data_utils.Subset(train_dataset, indices)
test_dataset = data_utils.Subset(test_dataset, indices)


model = BertForSequenceClassification.from_pretrained(
    PRE_TRAINED_MODEL_NAME,
    num_labels=len(df["Target"].unique()),
    output_attentions=False,
    output_hidden_states=False,
)


weight_decay = 0.5
learning_rate = 5.0e-5
batch_size = 32

training_args = TrainingArguments(
    seed=42,
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    report_to=["comet_ml"],
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
