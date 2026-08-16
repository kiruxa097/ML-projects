import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import numpy as np

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    train=True,
    transform=transform,
    root='data',
    download=True
)

test_dataset = datasets.MNIST(
    train=False,
    root='data',
    download=True,
    transform=transform
)

train_loader = DataLoader (
    train_dataset,
    batch_size=100,
    shuffle=True
)

test_loader = DataLoader (
    test_dataset,
    batch_size=100
)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Conv2d(1, 16, 3)
        self.layer2 = nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(16, 32, 3)
        self.layer4 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.layer5 = nn.Linear(800, 128)
        self.relu1 = nn.ReLU()
        self.layer6 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.layer7 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.layer8 = nn.Linear(32, 10)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flat(x)
        x = self.layer5(x)
        x = self.relu1(x)
        x = self.layer6(x)
        x = self.relu2(x)
        x = self.layer7(x)
        x = self.relu3(x)
        x = self.layer8(x)     

        return x

model = MyModel()

loss_entry = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

for epoch in range(20):
    for X, y in train_loader:
        output = model(X)
        optimizer.zero_grad()
        loss = loss_entry(output, y)
        loss.backward()
        optimizer.step()

x_test1, y_test1 = test_dataset[0]
x_test1 = x_test1.unsqueeze(0)
print(model(x_test1))
print(y_test1)