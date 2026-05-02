from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()

kf = KFold(n_splits=5, shuffle=True, random_state=45)

results = []

for train_idx, test_idx in kf.split(iris.data, iris.target):
    X_train, X_test = iris.data[train_idx], iris.data[test_idx]
    y_train, y_test = iris.target[train_idx], iris.target[test_idx]

    model = DecisionTreeClassifier(max_depth=3, random_state=45)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    accuracy = accuracy_score(y_test, predict)

    results.append(accuracy)
    print(f"Fold {len(results)} accuracy: {accuracy:.3f}")

print(f"\nСредняя точность: {np.mean(results):.3f} (+- {np.std(results):.3f})")