# Датасет
from sklearn.datasets import load_breast_cancer
# Модели 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Остальное 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np

# Загружаем датасет
bc = load_breast_cancer()

# Для удобства обозначаем признаки и классы (результаты относительно признаков)
X = bc.data
y = bc.target

# Делим данные на обучение и на проверку корректности работы моделей
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=765)


# Модели на произвольных характеристиках 
print("РЕЗУЛЬТАТЫ МОДЕЛЕЙ НА ПРОИЗВОЛЬНЫХ ХАРАКТЕРИСТИКАХ МОДЕЛЕЙ")
# Первая модель (Обычное дерево)
model1 = DecisionTreeClassifier(max_depth=20, random_state=97) # Сама модель
model1.fit(X_train, y_train) # Обучаем модель
pred1 = model1.predict(X_test) # Предсказываем результаты на тестовых данных
res1 = accuracy_score(y_test, pred1) # Рассчитываем точность модели

# Результаты первой модели
print(f"======== Результаты первой модели (DecisionTree) ========")
print(f"Точность модели: {res1*100:.3f}%")

# Вторая модель (n-neighbors)
model2 = KNeighborsClassifier(n_neighbors=10) # Сама модель
model2.fit(X_train, y_train) # Обучаем модель
pred2 = model2.predict(X_test) # Предсказываем результаты на тестовых данных
res2 = accuracy_score(y_test, pred2) # Рассчитываем точность модели

# Результаты второй модели
print(f"======== Результаты второй модели (KNeighbors) ========")
print(f"Точность модели: {res2*100:.3f}%")

# Третья модель (Random Forest)
model3 = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=97) # Сама модель
model3.fit(X_train, y_train) # Обучаем модель

# Получим название и сами признаки, чтобы потом определить их важность
importances = model3.feature_importances_
features = bc.feature_names
inc = np.argsort(importances)[::-1]

pred3 = model3.predict(X_test) # Предсказываем результаты на тестовых данных
res3 = accuracy_score(y_test, pred3) # Рассчитываем точность модели

# Результаты третей модели
print(f"======== Результаты третей модели (RandomForest) ========")
print(f"Точность модели: {res3*100:.3f}%")
print("======= Важность признаков =======")
for i in range(len(features)):
    print(f"Название признака: {features[inc[i]]}, важность: {importances[inc[i]]}")

# Четвертая модель (LogisticRegression)
model4 = LogisticRegression() # Сама модель
model4.fit(X_train, y_train) # Обучаем модель
pred4 = model4.predict(X_test) # Предсказываем результаты на тестовых данных
res4 = accuracy_score(y_test, pred4) # Рассчитываем точность модели

# Результаты четвертой модели
print(f"======== Результаты четвертой модели (LogisticRegression) ========")
print(f"Точность модели: {res4*100:.3f}%")
print()

# Теперь найдем лучшие точности моделей на разных параметрах и узнаем их параметры
print("ЛУЧШИЕ РЕЗУЛЬТАТЫ И ИХ ПАРАМЕТРЫ:")

# Pipeline для первой модели
pipe1 = Pipeline ([
    ("scaler", StandardScaler()),
    ("tree", DecisionTreeClassifier())
])

# Параметры для первой модели
gp1 = {
    "tree__max_depth" : [3, 5, 10, 15, 20, 30, 50],
    "tree__random_state" : [97],
}

# GridSearch для первой модели
grid1 = GridSearchCV(pipe1, gp1, cv=5)

grid1.fit(X_train, y_train) # Обучаем модель

# Лучшие параметры и лучшая точность при этих параметрах (первая модель)
print(f"======== Результаты первой модели (DecisionTree) ========")
print(f"Лучшие параметры: {grid1.best_params_}")
print(f"Лучшая точность: {grid1.best_score_*100:.3f}%")

# Pipeline для второй модели
pipe2 = Pipeline ([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Параметры для второй модели
gp2 = {
    "knn__n_neighbors" : [5, 10, 20, 30, 50],
}

# GridSearch для второй модели
grid2 = GridSearchCV(pipe2, gp2, cv=5)

grid2.fit(X_train, y_train) # Обучаем модель

# Лучшие параметры и лучшая точность при этих параметрах (вторая модель)
print(f"======== Результаты второй модели (KNeighbors) ========")
print(f"Лучшие параметры: {grid2.best_params_}")
print(f"Лучшая точность: {grid2.best_score_*100:.3f}%")

# Pipeline для третей модели
pipe3 = Pipeline ([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier())
])

# Параметры для третей модели
gp3 = {
    "rf__n_estimators" : [50, 100, 150, 200],
    "rf__max_depth" : [3, 5, 10, 15, 20, 30, 50],
    "rf__random_state" : [97],
}

# GridSearch для третей модели
grid3 = GridSearchCV(pipe3, gp3, cv=5)

grid3.fit(X_train, y_train) # Обучаем модель

# Лучшие параметры и лучшая точность при этих параметрах (третья модель)
print(f"======== Результаты третей модели (Random Forest) ========")
print(f"Лучшие параметры: {grid3.best_params_}")
print(f"Лучшая точность: {grid3.best_score_*100:.3f}%")

# Pipeline для четвертой модели
pipe4 = Pipeline ([
    ("scaler", StandardScaler()),
    ("lg", LogisticRegression())
])

# Параметры для четвертой модели
gp4 = {
    "lg__C" : [0.1, 1, 10, 1000],
    "lg__penalty" : ['l1', 'l2'],
    "lg__max_iter" : [100, 1000],
}

# GridSearch для четвертой модели
grid4 = GridSearchCV(pipe4, gp4, cv=5)

grid4.fit(X_train, y_train) # Обучаем модель

# Лучшие параметры и лучшая точность при этих параметрах (четвертая модель)
print(f"======== Результаты четвертой модели (LogisticRegression) ========")
print(f"Лучшие параметры: {grid4.best_params_}")
print(f"Лучшая точность: {grid4.best_score_*100:.3f}%")

# Найдем самую точную модель
models = {
    "DecisionTree" : grid1,
    "KNeighbors" : grid2,
    "RandomForest" : grid3,
    "LogisticRegression" : grid4,
}

name_best_models = ""
best_scor = 0
for name, par in models.items():
    if par.best_score_*100 > best_scor:
        best_scor = par.best_score_*100
        name_best_models = name

print()
# Вывод по всей работе
print("========= КРАТКИЙ ВЫВОД ==========")
print(f"Лучшая модель: {name_best_models}, ее точность составляет: {best_scor:.3f}")