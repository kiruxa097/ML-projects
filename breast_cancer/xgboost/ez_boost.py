import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

breast = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(breast.data, breast.target, test_size=0.2, random_state=67)

model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)

model.fit(X_train, y_train)
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)*100

print(f"Точность модели: {acc:.1f}%")