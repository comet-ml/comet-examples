# coding: utf-8
import comet_ml

import composer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from composer.loggers import CometMLLogger
from composer.models import ComposerClassifier
from torchvision import datasets, transforms

comet_ml.init(project_name="comet-example-mosaicml-getting-started")
torch.manual_seed(42)  # For replicability

data_directory = "./data"

# Normalization constants
mean = (0.507, 0.487, 0.441)
std = (0.267, 0.256, 0.276)

batch_size = 1024

cifar10_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)]
)

train_dataset = datasets.CIFAR10(
    data_directory, train=True, download=True, transform=cifar10_transforms
)
test_dataset = datasets.CIFAR10(
    data_directory, train=False, download=True, transform=cifar10_transforms
)

# Our train and test dataloaders are PyTorch DataLoader objects!
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)


class Block(nn.Module):
    """A ResNet block."""

    def __init__(self, f_in: int, f_out: int, downsample: bool = False):
        super(Block, self).__init__()

        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(
            f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(f_out)
        self.conv2 = nn.Conv2d(
            f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(f_out)
        self.relu = nn.ReLU(inplace=True)

        # No parameters for shortcut connections.
        if downsample or f_in != f_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(f_out),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x: torch.Tensor):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNetCIFAR(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    def __init__(self, outputs: int = 10):
        super(ResNetCIFAR, self).__init__()

        depth = 56
        width = 16
        num_blocks = (depth - 2) // 6

        plan = [(width, num_blocks), (2 * width, num_blocks), (4 * width, num_blocks)]

        self.num_classes = outputs

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(
            3, current_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(current_filters)
        self.relu = nn.ReLU(inplace=True)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], outputs)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        out = self.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = ComposerClassifier(module=ResNetCIFAR(), num_classes=10)

optimizer = composer.optim.DecoupledSGDW(
    model.parameters(),  # Model parameters to update
    lr=0.05,  # Peak learning rate
    momentum=0.9,
    weight_decay=2.0e-3,
)

lr_scheduler = composer.optim.LinearWithWarmupScheduler(
    t_warmup="1ep",  # Warm up over 1 epoch
    alpha_i=1.0,  # Flat LR schedule achieved by having alpha_i == alpha_f
    alpha_f=1.0,
)

logger_for_baseline = CometMLLogger()

train_epochs = "3ep"
device = "gpu" if torch.cuda.is_available() else "cpu"

trainer = composer.trainer.Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    max_duration=train_epochs,
    optimizers=optimizer,
    schedulers=lr_scheduler,
    device=device,
    loggers=logger_for_baseline,
)

trainer.fit()  # <-- Your training loop in action!
