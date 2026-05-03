from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
# Делаем искуственную проблему
X[:, 0] = X[:, 0] * 1000

# Сначала делаем модель без масштабирования
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predict = model.predict(X_test)
accuracy = accuracy_score(y_test, predict)


# Теперь сначала масштабируем, а после обучаем модель
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaler = scaler.transform(X_train)
X_test_scaler = scaler.transform(X_test)
# И теперь обучаем модель на новых данных
model_scaler = KNeighborsClassifier(n_neighbors=5)
model_scaler.fit(X_train_scaler, y_train)
predict2 = model_scaler.predict(X_test_scaler)
accuracy_scaler = accuracy_score(y_test, predict2)

print(f"KNN без масштабирования: {accuracy:.3f}")
print(f"KNN С масштабированием: {accuracy_scaler:.3f}")