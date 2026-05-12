from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Исходные данные
data = pd.DataFrame({
    'city': ['Moscow', 'SPb', 'Kazan', 'Moscow', 'SPb', 'Kazan'],
    'price': [100, 150, 120, 110, 160, 130],
    'rooms': [2, 3, 2, 3, 2, 3]
})
y = [0, 1, 0, 1, 0, 1]

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=45)

# Определяем числовые и категориальные колонки 
numeric_cols = ["price", "rooms"]
categorical_cols = ["city"]

preprocessor = ColumnTransformer ([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first"), categorical_cols),
])

pipe = Pipeline ([
    ("pr", preprocessor),
    ("rf", RandomForestClassifier()),
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print(f"Точность модели: {accuracy_score(y_test, pred)*100:.1f}%")