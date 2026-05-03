from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=68)

model = KNeighborsClassifier()

param_grid = {
    'n_neighbors':[1,3,5,7,9,11,13,15],
    'weights':["uniform", "distance"],
}

grid_s = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0
)

grid_s.fit(X_train, y_train)

print("======= РЕЗУЛЬТАТЫ GRIDSEARCHCV =======")
print(f"Лучшие параметры: {grid_s.best_params_}")
print(f"Лучшая точность (кросс-валидация): {grid_s.best_score_:.3f}")
print()

# Проверяем на лучшей модели
best_model = grid_s.best_estimator_
predict = best_model.predict(X_test)
best_acc = accuracy_score(y_test, predict)
print("======= РЕЗУЛЬТАТЫ НА ЛУЧШЕЙ ВЫБОРКИ =======")
print(f"Точность на тесте: {best_acc:.3f}")