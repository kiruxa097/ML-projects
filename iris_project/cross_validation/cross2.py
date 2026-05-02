from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()

X, y = iris.data, iris.target
model = DecisionTreeClassifier(max_depth=5, random_state=12)

print("=========== Единичные показатели ===========")

# Одно разбиение (у 3 разных проверяем результаты)
for rs in range(42,47):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    accuracy = accuracy_score(y_test, predict)
    print(f"random_state={rs}, accuracy={accuracy:.3f}")

# Кросс-валидация (более точный ответ, если std большое, то модель нестабильная)
cross = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print()
print("=========== Кросс-валидация ===========")
print(f"Средняя точность: {cross.mean():.3f} (+- {cross.std():.3f})")
print(f"Разница между максимальным показателем и минимальным: {(cross.max() - cross.min()):.3f}")
print()