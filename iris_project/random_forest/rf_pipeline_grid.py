from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1313)

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=4567)

pipe = Pipeline ([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier())
])

param_grid = {
    "rf__n_estimators" : [100, 200, 300],
    "rf__max_depth" : [3, 5, 10]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

print(f"Лучшие параметры: {grid.best_params_}")
print(f"Лучшая точность: {grid.best_score_:.3f}")
print(f"Точность на тесте: {accuracy_score(y_test, grid.predict(X_test)):.3f}")