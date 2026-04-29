from ml_models import my_ml
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from random import randint

# Датасет "Ирис"
iris = load_iris()

# Рандомное расположение входных данных
rand = randint(1,10000)

# Делим данные для обучения с учителем (80% на обучения, 20% на тест)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=rand)


# Массивы для результатов эксперимента
mass1, mass2 = [], []

for i in range(1, 120):
    r1, r2 = my_ml(X_train, X_test, y_train, y_test, i)
    mass1.append(r1)
    mass2.append(r2)

ans1 = 0
ans2 = 0
ans3 = 0

for i in range(0,118):
    if mass1[i] == mass2[i]:
        ans3 += 1
    elif mass1[i] > mass2[i]:
        ans1 += 1
    else:
        ans2 += 1
print(f"Модели показали одинаковую точность: {ans3} раз.")
print(f"Модель дерева показала более точно: {ans1} раз.")
print(f"Модель k-соседей показала более точно: {ans2} раз.")
print(f"Точность при глубине 25 у модели дерева: {mass1[24]*100:.1f}%")
print(f"Точность при глубине 25 у модели k-соседей: {mass2[24]*100:.1f}%")
print(f"Точность при глубине 50 у модели дерева: {mass1[49]*100:.1f}%")
print(f"Точность при глубине 50 у модели k-соседей: {mass2[49]*100:.1f}%")
print(f"Точность при глубине 75 у модели дерева: {mass1[74]*100:.1f}%")
print(f"Точность при глубине 75 у модели k-соседей: {mass2[74]*100:.1f}%")
print(f"Точность при глубине 100 у модели дерева: {mass1[99]*100:.1f}%")
print(f"Точность при глубине 100 у модели k-соседей: {mass2[99]*100:.1f}%")
print(f"Точность при глубине 119 у модели дерева: {mass1[118]*100:.1f}%")
print(f"Точность при глубине 119 у модели k-соседей: {mass2[118]*100:.1f}%")
print(f"Рандомное расположение входных данных: {rand}")