import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt

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

train_d_t, train_d_v = random_split(train_dataset, [0.7, 0.3])

train_dataloader = DataLoader(
    train_d_t,
    shuffle=True,
    batch_size=64
)

validation_dataloader = DataLoader(
    train_d_v,
    shuffle=True,
    batch_size=64
)

test_dataloader = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=64
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

loss_on_train = []
loss_on_val = []

for epoch in range(20):

    loss_on_steps = 0
    k = 0

    model.train() # Переводим модель в тренировочный режим, считаем градиенты и т.д

    for X_train, y_train in train_dataloader:
        output = model(X_train)
        optimizer.zero_grad()
        loss = loss_entry(output, y_train)
        loss.backward()
        optimizer.step()

        loss_on_steps += loss.item()
        k += 1

    model.eval()

    loss_on_steps1 = 0
    k1 = 0

    for X_val, y_val in validation_dataloader:
        output = model(X_val)
        loss = loss_entry(output, y_val)
        loss_on_steps1 += loss.item()
        k1 += 1

    loss_on_train.append(loss_on_steps/k)
    loss_on_val.append(loss_on_steps1/k1)

# Отобразим графики значений функции потерь и эпоху, чтобы посмотреть, где графики начинают расходиться, дабы пересечь переобучение 
plt.plot(loss_on_train)
plt.plot(loss_on_val)
plt.grid()
plt.show()

loss_on_test = 0
k2 = 0
model.eval()

for X_test, y_test in test_dataloader:
    output = model(X_test)
    loss = loss_entry(output, y_test)
    loss_on_test += loss
    k2 += 1

print(f'Результат функции потерь: {loss_on_test/k2}')

