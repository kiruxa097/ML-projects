from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

houses = fetch_california_housing()

X, y = houses.data, houses.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=92)

model = LinearRegression()
model.fit(X_train, y_train)
predict = model.predict(X_test)

print("======== Результаты =========")
print(f"R²: {r2_score(y_test, predict):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predict)):.4f}")