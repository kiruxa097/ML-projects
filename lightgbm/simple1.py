import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

bc = load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

pipe = Pipeline ([
    ("lgbm", lgb.LGBMClassifier(random_state=12, verbose=-1))
])

param_g = {
    "lgbm__n_estimators" : [100,200,500,1000],
    "lgbm__learning_rate" : [0.01, 0.05, 0.1, 0.2, 0.5, 1],
    "lgbm__num_leaves" : [31, 63, 127]
}

grid = GridSearchCV(estimator=pipe, cv=5, param_grid=param_g)

grid.fit(X_train, y_train)

print(f"Лучшя точность модели: {grid.best_score_*100:.2f}%")
print(f"Лучшие параметры для модели: {grid.best_params_}")