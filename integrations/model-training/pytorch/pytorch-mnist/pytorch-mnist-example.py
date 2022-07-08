# coding: utf-8
from comet_ml import Experiment, init

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

hyper_params = {
    "sequence_length": 28,
    "input_size": 28,
    "hidden_size": 128,
    "num_layers": 2,
    "num_classes": 10,
    "batch_size": 100,
    "num_epochs": 3,
    "learning_rate": 0.01,
}


# Login to Comet if needed
init()

experiment = Experiment(project_name="comet-example-pytorch-mnist")
experiment.log_parameters(hyper_params)


# MNIST Dataset
train_dataset = dsets.MNIST(
    root="./data/", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = dsets.MNIST(root="./data/", train=False, transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=hyper_params["batch_size"], shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=hyper_params["batch_size"], shuffle=False
)


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

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=hyper_params["learning_rate"])

# Train the Model
total_steps = len(train_dataset) // hyper_params["batch_size"]

with experiment.train():

    step = 0
    for epoch in range(hyper_params["num_epochs"]):
        experiment.log_current_epoch(epoch)
        correct = 0
        total = 0
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

            # Log accuracy to Comet.ml
            experiment.log_metric("accuracy", 100 * correct / total, step=step)
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


with experiment.test():
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(
            images.view(-1, hyper_params["sequence_length"], hyper_params["input_size"])
        )
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += float((predicted == labels).sum())

    experiment.log_metric("accuracy", correct / total)
    print(
        "Test Accuracy of the model on the 10000 test images: %d %%"
        % (100 * correct / total)
    )
