from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris



def my_ml(X_train, X_test, y_train, y_test, k):

    # Дерево
    t1 = DecisionTreeClassifier(max_depth=k)

    # Соседи
    t2 = KNeighborsClassifier(n_neighbors=k)

    # Обучаем обе модели
    t1.fit(X_train, y_train)
    t2.fit(X_train, y_train)

    # Предсказываем результаты
    res1 = t1.predict(X_test)
    res2 = t2.predict(X_test)

    # Рассчитываем точность моделей
    acc1 = accuracy_score(res1, y_test)
    acc2 = accuracy_score(res2, y_test)

    return acc1, acc2