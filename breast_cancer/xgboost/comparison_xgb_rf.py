import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

breast = load_breast_cancer()
X = breast.data
y = breast.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

model_rf = RandomForestClassifier(n_estimators=150, max_depth=3)
model_xgb = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1)

model_rf.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)

pred1 = model_rf.predict(X_test)
pred2 = model_xgb.predict(X_test)

acc1 = accuracy_score(y_test, pred1)
acc2 = accuracy_score(y_test, pred2)

print(f"Точность модели (RandomForestClassifier): {acc1*100:.2f}%")
print(f"Точность модели (XGBClassifier): {acc2*100:.2f}%")