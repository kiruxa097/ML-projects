from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=57)

# Обучаем модель
model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=2424)
model.fit(X_train, y_train)

# Получаем значения и названия признаков
importances = model.feature_importances_
features = iris.feature_names

inc = np.argsort(importances)[::-1]

print("======= Важность признаков =======")
for i in range(len(features)):
    print(f"Название признака: {features[inc[i]]}, важность: {importances[inc[i]]}")
