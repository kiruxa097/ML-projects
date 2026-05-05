from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=75)

# Обычное одно дерево
model1 = DecisionTreeClassifier(max_depth=3, random_state=224)
model1.fit(X_train, y_train)
predict1 = model1.predict(X_test)
accuracy1 = accuracy_score(y_test, predict1)

# RandomForest (тут 200 деревьев)
model2 = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=224)
model2.fit(X_train, y_train)
predict2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test, predict2)

print(f"Точность обычного 1 дерева: {accuracy1:.3f}")
print(f"Точность RandomForest: {accuracy2:.3f}")