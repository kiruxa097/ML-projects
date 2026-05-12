import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Загружаем датасет titanic 
titanic = sns.load_dataset("titanic")

# Распределим часть данных из датасета
X = titanic[["pclass", "sex", "age", "fare", "embarked"]]
y = titanic["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Разделим данные на категории
numeric_cols = ["age", "fare"]
categorical_cols = ["sex", "embarked", "pclass"]

preprocessor = ColumnTransformer ([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("enc", OneHotEncoder(drop="first", sparse_output=False))
    ]), categorical_cols)
])

pipe = Pipeline ([
    ("prep", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=32))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print(f"Точность модели: {accuracy_score(y_test, pred)*100:.1f}%")