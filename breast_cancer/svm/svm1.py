from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

bc = load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

pipe = Pipeline ([
    ("scaler", StandardScaler()),
    ("svc", SVC())
])

grid_p = {
    "svc__C" : [0.1, 1, 10, 100],
    "svc__kernel" : ["rbf", "poly"],
    "svc__gamma" : ["scale", "auto", 0.1, 1, 10]
}

grid = GridSearchCV(pipe, grid_p, cv=5)

grid.fit(X_train, y_train)

print("======== РЕЗУЛЬТАТЫ МОДЕЛИ SVC ========")
print(f"Лучшие параметры: {grid.best_params_}")
print(f"Лучший результат: {grid.best_score_*100:.3f}%")