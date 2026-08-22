import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    train=True,
    download=True,
    root='data',
    transform=transform
)

test_dataset = datasets.MNIST(
    train=False,
    download=True,
    root='data',
    transform=transform
)

train_data, val_data = random_split(train_dataset, [0.7, 0.3])

train_dataloader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

val_dataloader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=True
)

test_datalloader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

func_loss = nn.CrossEntropyLoss()

train_loss, val_loss = [], []

for epoch in range(10):

    k1, loss1 = 0, 0
    k2, loss2 = 0, 0

    model.train()

    for X_train, y_train in train_dataloader:

        optimizer.zero_grad()
        output = model(X_train)
        loss = func_loss(output, y_train)
        loss.backward()
        optimizer.step()

        loss1 += loss.item()
        k1 += 1

    model.eval()

    with torch.no_grad():
        for X_val, y_val in val_dataloader:

            output2 = model(X_val)
            loss = func_loss(output2, y_val)

            k2 += 1
            loss2 += loss.item()

    train_loss.append(loss1/k1)
    val_loss.append(loss2/k2)

plt.plot(train_loss)
plt.plot(val_loss)
plt.grid()
plt.show()
