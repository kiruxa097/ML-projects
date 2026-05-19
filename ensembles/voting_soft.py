from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

bc = load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

model1 = RandomForestClassifier(n_estimators=150)
model2 = XGBClassifier(n_estimators=150, learning_rate=0.1)
model3 = LGBMClassifier(n_estimators=150, learning_rate=0.1)

voting_soft = VotingClassifier(
    estimators=[
        ("rf", model1),
        ("xgb", model2),
        ("lgb", model3)
    ],
    voting='soft'
)

voting_soft.fit(X_train, y_train)

predict = voting_soft.predict(X_test)

print(f"Итоговая точность(взвешенные вероятности моделей): {accuracy_score(y_test, predict) * 100 :.2f}%")