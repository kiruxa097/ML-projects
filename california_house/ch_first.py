from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

ch = fetch_california_housing()

X= ch.data

print("=" * 50)
print("ДАННЫЕ: Калифорнийские дома")
print("=" * 50)
print(f"Количество объектов: {X.shape[0]}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Признаки: {ch.feature_names}")
print(f"Целевая переменная: цена дома (в hundreds of thousands $)")