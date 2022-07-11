# coding: utf-8
import random

from comet_ml import ConfusionMatrix, Experiment, init

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

hyper_params = {
    "sequence_length": 28,
    "input_size": 28,
    "hidden_size": 128,
    "num_layers": 3,
    "num_classes": 10,
    "batch_size": 100,
    "num_epochs": 2,
    "learning_rate": 0.02,
}

# Login to Comet if needed
init()


experiment = Experiment(project_name="comet-example-pytorch-rich-logging")

experiment.add_tag("pytorch")

# Log hyperparameters to Comet
experiment.log_parameters(hyper_params)

# MNIST Dataset
train_dataset = dsets.MNIST(
    root="./data/", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = dsets.MNIST(root="./data/", train=False, transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=hyper_params["batch_size"], shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=hyper_params["batch_size"], shuffle=False
)


# Log dataset sample images to Comet
num_samples = len(train_dataset)
for _ in range(10):
    value = random.randint(0, num_samples)
    tmp, _ = train_dataset[value]
    img = tmp.numpy()[0]
    experiment.log_image(img, name="groundtruth:{}".format(_))


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


rnn = RNN(
    hyper_params["input_size"],
    hyper_params["hidden_size"],
    hyper_params["num_layers"],
    hyper_params["num_classes"],
)

experiment.set_model_graph(str(rnn))

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=hyper_params["learning_rate"])


def train_index_to_example(index):
    tmp, _ = train_dataset[index]
    img = tmp.numpy()[0]
    data = experiment.log_image(img, name="train_%d.png" % index)

    if data is None:
        return None

    return {"sample": str(index), "assetId": data["imageId"]}


def test_index_to_example(index):
    tmp, _ = test_dataset[index]
    img = tmp.numpy()[0]
    data = experiment.log_image(img, name="test_%d.png" % index)

    if data is None:
        return None

    return {"sample": str(index), "assetId": data["imageId"]}


def onehot(i):
    v = [0] * 10
    v[i] = 1
    return v


# Make one to use repeatedly, to re-use examples where possible:
confusion_matrix = ConfusionMatrix(index_to_example_function=train_index_to_example)

# Train the Model
total_steps = len(train_dataset) // hyper_params["batch_size"]
with experiment.train():

    print("Logging weights as histogram (before training)...")
    # Log model weights
    weights = []
    for name in rnn.named_parameters():
        if "weight" in name[0]:
            weights.extend(name[1].detach().numpy().tolist())
    experiment.log_histogram_3d(weights, step=0)

    step = 0
    for epoch in range(hyper_params["num_epochs"]):
        experiment.log_current_epoch(epoch)
        correct = 0
        total = 0

        epoch_predictions = None
        epoch_targets = None

        for i, (images, labels) in enumerate(train_loader):
            images = Variable(
                images.view(
                    -1, hyper_params["sequence_length"], hyper_params["input_size"]
                )
            )
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute train accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += float((predicted == labels.data).sum())

            if epoch_predictions is not None:
                epoch_predictions = np.concatenate(
                    (epoch_predictions, outputs.data.numpy())
                )
            else:
                epoch_predictions = outputs.data.numpy()

            if epoch_targets is not None:
                epoch_targets = np.concatenate(
                    (epoch_targets, np.array([onehot(v) for v in labels]))
                )
            else:
                epoch_targets = np.array([onehot(v) for v in labels])

            # Log accuracy to Comet.ml
            experiment.log_metric("accuracy", correct / total, step=step)
            step += 1

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [%d/%d], Step [%d/%d], Loss: %.4f"
                    % (
                        epoch + 1,
                        hyper_params["num_epochs"],
                        i + 1,
                        total_steps,
                        loss.data.item(),
                    )
                )

        # At end of epoch:
        print("Computing confusion matrix and uploading samples...")
        confusion_matrix.compute_matrix(epoch_targets, epoch_predictions)
        experiment.log_confusion_matrix(
            matrix=confusion_matrix,
            title="Train Confusion Matrix, Epoch #%s" % (epoch + 1,),
            file_name="train-confusion-matrix-%03d.json" % (epoch + 1),
        )

        print("Logging weights as histogram...")
        # Log model weights
        weights = []
        for name in rnn.named_parameters():
            if "weight" in name[0]:
                weights.extend(name[1].detach().numpy().tolist())
        experiment.log_histogram_3d(weights, step=epoch + 1)


with experiment.test():
    # Test the Model
    correct = 0
    total = 0

    test_predictions = None
    test_targets = None
    for images, labels in test_loader:
        images = Variable(
            images.view(-1, hyper_params["sequence_length"], hyper_params["input_size"])
        )
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += float((predicted == labels).sum())

        if test_predictions is None:
            test_predictions = np.array([onehot(v) for v in labels])
        else:
            test_predictions = np.concatenate(
                (test_predictions, np.array([onehot(v) for v in labels]))
            )

        if test_targets is None:
            test_targets = outputs.data.numpy()
        else:
            test_targets = np.concatenate((test_targets, outputs.data.numpy()))

    experiment.log_confusion_matrix(
        test_targets,
        test_predictions,
        title="Test Confusion Matrix",
        file_name="test-confusion-matrix.json",
        index_to_example_function=test_index_to_example,
    )

    experiment.log_metric("accuracy", correct / total)
    print(
        "Test Accuracy of the model on the 10000 test images: %d %%"
        % (100 * correct / total)
    )
