from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Загружаем датасет : ирис
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df.iloc[0:10, 0] = np.nan # Делаем пропуски в первом признаке
df.iloc[5:15, 2] = np.nan # Делаем пропуски в третьем признаке

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=88)

model = RandomForestClassifier()

pipe = Pipeline ([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=55))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
ans = accuracy_score(y_test, pred) * 100
print(f"Точность модели: {ans:.3f}%")