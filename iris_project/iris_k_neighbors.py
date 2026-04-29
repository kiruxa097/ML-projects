from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()

# Разделяем данные на тренировочную (обучающую) модель и на тестовую, с помощью которой проверим точность модели
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=32)

# Сравниваем результаты с соседями
knn = KNeighborsClassifier(n_neighbors=3)

# Обучаем модель (k neighbors)
knn.fit(X_train, y_train)

# Предугадываем результат, основываясь на тренировочных результатах модели
predict = knn.predict(X_test)

# Точность результатов
accuracy = accuracy_score(y_test, predict)

print(f"Точность модели: {accuracy*100:.1f}%")