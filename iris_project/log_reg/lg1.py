from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()
X = iris.data[:, 2:]
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=76)

m_lg = LogisticRegression()
m_lg.fit(X_train, y_train)

pred = m_lg.predict(X_test)
acc = accuracy_score(y_test, pred)
ver = m_lg.predict_proba(X_test)

print(f"Точность модели: {acc*100:.1f}%")
print(f"Вероятность первых 3 ирисов из тестовых данных: \n{ver[:3]}")