import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

tit = sns.load_dataset('titanic')

X = tit[["pclass", "sex", "age", "fare", "embarked"]]
y = tit["survived"]

X['embarked'] = X['embarked'].fillna('S')
X['sex'] = X['sex'].fillna('unknown')
X['pclass'] = X['pclass'].fillna(3).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=63)

numeric_cols = ["age", "fare"]
categorical_cols = ["sex", "embarked", "pclass"]

model = CatBoostClassifier(random_state=33, verbose=0, allow_writing_files=False)

param_grid = {
    "iterations" : [100, 300],
    "depth" : [3, 5],
    "learning_rate" : [0.05, 0.1],
    "l2_leaf_reg" : [3, 7],
}

grid = GridSearchCV(estimator=model, cv=5, param_grid=param_grid)

grid.fit(X_train, y_train, cat_features=categorical_cols)

print(f"Лучший результат CatBoost: {grid.best_score_*100:.2f}%")
print(f"Лучшие параметры CatBoost: {grid.best_params_}")