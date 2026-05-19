from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

bc = load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

model1 = RandomForestClassifier(n_estimators=150)
model2 = LGBMClassifier(n_estimators=150, learning_rate=0.1, verbose=-1)

stacking = StackingClassifier(
    estimators= [
        ("rf", model1),
        ("lgb", model2)
    ],
    final_estimator=XGBClassifier(n_estimators=150, learning_rate=0.1),
    cv=5
)

stacking.fit(X_train, y_train)

pred = stacking.predict(X_test)

print(f"Точнсть модели: {accuracy_score(y_test, pred)*100:.2f}%")