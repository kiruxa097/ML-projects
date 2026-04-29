from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()

# Делим данные для обучения с учителем (80% на обучения, 20% на тест)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=67)

print(f"Обучающая выборка: {X_train.shape[0]} цветков")
print(f"Тестовая выборка: {X_test.shape[0]} цветков")

model = DecisionTreeClassifier(max_depth=3) 

# Обучаем дерево
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Точность модели: {accuracy*100:.1f}%")
