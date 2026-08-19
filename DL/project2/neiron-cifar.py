import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root='data',
    download=True,
    transform=transform,
    train=True
)

test_dataset = datasets.CIFAR10(
    root='data',
    download=True,
    train=False,
    transform=transform
)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=100
)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Conv2d(3, 48, 3)
        self.pol1 = nn.MaxPool2d(2)
        self.layer2 = nn.Conv2d(48, 96, 4)
        self.pol2 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.layer3 = nn.Linear(3456, 256)
        self.relu1 = nn.ReLU()
        self.layer4 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.layer5 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.layer6 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        self.layer7 = nn.Linear(32, 10)

    def forward(self, x):

        x = self.layer1(x)
        x = self.pol1(x)
        x = self.layer2(x)
        x = self.pol2(x)
        x = self.flat(x)
        x = self.layer3(x)
        x = self.relu1(x)
        x = self.layer4(x)
        x = self.relu2(x)
        x = self.layer5(x)
        x = self.relu3(x)
        x = self.layer6(x)
        x = self.relu4(x)
        x = self.layer7(x)

        return x

model = MyModel()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

for epoch in range(20):
    for X_train, y_train in train_dataloader:

        output = model(X_train)

        optimizer.zero_grad()

        loss = loss_fn(output, y_train)
        loss.backward()

        optimizer.step()

correct_ans = 0
all_test = len(test_dataset)

for X_test, y_test in test_dataset:
    X_test = X_test.unsqueeze(0)
    pred = model(X_test)
    if pred.argmax(dim=1) == y_test: correct_ans += 1

print('Точность модели: ', correct_ans/all_test*100,'%')