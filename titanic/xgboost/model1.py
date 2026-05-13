from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

# Загружаем датасет - титаник
titanic = sns.load_dataset("titanic")

# Распределим часть данных из датасета
X = titanic[["pclass", "sex", "age", "fare", "embarked"]]
y = titanic["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Разделим данные на категории
numeric_cols = ["age", "fare"]
categorical_cols = ["sex", "embarked", "pclass"]

# К разным категориям применяем разные методы, числовые - вставляем пропуски и масштабируем, а категориальные - вставляем пропуски и делаем из категорий - число
preprocesor = ColumnTransformer ([
    (
        "num", Pipeline ([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
    ]), numeric_cols
    ),
    (
        "cat", Pipeline ([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(drop="first", sparse_output=False)),
    ]), categorical_cols
    )
])

# Pipeline
pipe = Pipeline ([
    ("prep", preprocesor),
    ("xg", xgb.XGBClassifier())
])

# Параметры для перебора в GridSearch
param_grid = {
    "xg__n_estimators" : [100, 150, 200, 250],
    "xg__max_depth" : [3, 5, 7, 10],
    "xg__learning_rate" : [0.1, 0.3, 0.5, 1],
}

# Применяем GridSearch для модели xgb
model_xgb = GridSearchCV (
    estimator=pipe,
    cv=5,
    param_grid=param_grid,
)

# Обучаем модель
model_xgb.fit(X_train, y_train)

# Вывод результатов
print(f"Лучшая точность модели: {model_xgb.best_score_*100:.2f}%")
print(f"Лучшие параметры для модели: {model_xgb.best_params_}")