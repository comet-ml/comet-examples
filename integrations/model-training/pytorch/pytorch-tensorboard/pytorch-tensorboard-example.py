# coding: utf-8
import comet_ml

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

hyper_params = {
    "batch_size": 4,
    "num_epochs": 1,
    "learning_rate": 0.01,
    "momentum": 0.01,
}


comet_ml.init()

experiment = comet_ml.Experiment(project_name="comet-example-pytorch-tensorboard")
experiment.log_parameters(hyper_params)

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# datasets
trainset = torchvision.datasets.FashionMNIST(
    "./data", download=True, train=True, transform=transform
)
testset = torchvision.datasets.FashionMNIST(
    "./data", download=True, train=False, transform=transform
)

# dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=hyper_params["batch_size"], shuffle=True, num_workers=2
)


testloader = torch.utils.data.DataLoader(
    testset, batch_size=hyper_params["batch_size"], shuffle=False, num_workers=2
)

# constant for classes
classes = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=hyper_params["learning_rate"],
    momentum=hyper_params["momentum"],
)

writer = SummaryWriter()


# helper functions


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure()
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]], probs[idx] * 100.0, classes[labels[idx]]
            ),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
        )
    fig.tight_layout()
    return fig


# Logs training images
# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# write to tensorboard
writer.add_image("four_fashion_mnist_images", img_grid)


# Log graph
writer.add_graph(net, images)


# Train the Model
total_steps = len(trainloader)

running_loss = 0.0
for epoch in range(hyper_params["num_epochs"]):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:  # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar(
                "training loss", running_loss / 1000, epoch * len(trainloader) + i
            )

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure(
                "predictions vs. actuals",
                plot_classes_preds(net, inputs, labels),
                global_step=epoch * len(trainloader) + i,
            )
            running_loss = 0.0

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
print("Finished Training")

writer.close()
