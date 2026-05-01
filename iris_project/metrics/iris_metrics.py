from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np

iris = load_iris()

# Делаем бинарную задачу (так как изначально матрики, которые будут использоваться в этой программе, определены для бинарной классификации)
y_binary = (iris.target == 0).astype(int)  # 1 если setosa, 0 если нет

X_train, X_test, y_train, y_test = train_test_split(iris.data, y_binary, test_size=0.2, random_state=42)

# Модель
mod = DecisionTreeClassifier(max_depth=3)
mod.fit(X_train, y_train)
pred = mod.predict(X_test)

# Находим элементы матрицы ошибок
tp, tn, fp, fn = confusion_matrix(y_test, pred).ravel()

# Построим матрицу ошибок 
print(f"-----Матрица ошибок-----")
print(f"|    TP={tp}  FN={fn}      |")
print(f"|    FP={fp}   TN={tn}       |")
print(f"------------------------")
print(f"Находим accuracy(делали ранее) и 3 новых метрики (Precision, Recall, F1-score)")
print(f"accuracy: {accuracy_score(y_test, pred)*100:.1f}")
print(f"precision: {precision_score(y_test, pred)*100:.1f}")
print(f"recall: {recall_score(y_test, pred)*100:.1f}")
print(f"f1-score: {f1_score(y_test, pred)*100:.1f}")