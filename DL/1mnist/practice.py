import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
import numpy as np

# Добавил свою картинку и превел ее к нужному shape 
image_test = Image.open("c:\\ML\\ML-projects\\DL\\1mnist\\converted_image.png").convert('L').resize((28,28))
pag = np.array(image_test, dtype=np.float32)
t_pag = torch.from_numpy(pag).unsqueeze(0).unsqueeze(0)

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
    shuffle=False,
    batch_size=100
)

class MNISTmodel(nn.Module):
    def __init__(self):
        super().__init__()

        # Слои "подбора" весов и bias
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128,64)
        self.layer3 = nn.Linear(64,10)
        # Убираем линейность (ReLU)
        self.relu = nn.ReLU()
        # Приводим к нужному shape 
        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)

        return x

model = MNISTmodel()

# Градиентный спуск 
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

# функция потерь
loss_fn = nn.CrossEntropyLoss()

# пробегаем 20 эпох с случайными разбиениями и в каждой по всем батчам, т.к сделал в каждом batch 100 картинок, их будет 600
for epoch in range(20):
    for x_train, y_train in train_loader:

        output = model(x_train) # Пробегаем все слои

        optimizer.zero_grad() # Обнуляем градиенты, чтобы потом посчитать новые
        loss = loss_fn(output, y_train) # Считаем "точность"
        loss.backward() # считаем градиенты 
        optimizer.step() # градиентный спуск

my_pred = model(t_pag) # передал свою картинку
print(my_pred.argmax()) # вывел индекс с макс значением - это и есть результат работы нейронной сети