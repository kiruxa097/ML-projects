from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np

iris = load_iris()
X = iris.data[:, 2: ]
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lg", LogisticRegression())
])

grid_p = {
    "lg__C" : [0.1, 1, 10, 1000],
    "lg__penalty" : ['l1', 'l2'],
    "lg__max_iter" : [100, 1000]
}

grid = GridSearchCV(pipe, grid_p, cv=5)
grid.fit(X_train, y_train)

print(f"Лучшие параметры: {grid.best_params_}")
print(f"Лучшая точность: {grid.best_score_:.3f}")