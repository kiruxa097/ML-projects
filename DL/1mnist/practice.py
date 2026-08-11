import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=100
)

test_loader = DataLoader(
    test_dataset,
    shuffle=True,
    batch_size=100
)

X_train, y_train = next(iter(train_loader)) # Первый батч
X_test, y_test = next(iter(test_loader))

y_train = y_train.long()
class MNISTmodel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(128,10)
        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

model = MNISTmodel()
output = model(X_train)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output, y_train)
print(loss)