from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

houses = fetch_california_housing()

X, y = houses.data, houses.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=145)

model = RandomForestRegressor(n_estimators=150, max_depth=5)

model.fit(X_train, y_train)

pred = model.predict(X_test)

# Выведем метрики (прежде вычислим их)

mae = mean_absolute_error(y_test, pred)

mse = mean_squared_error(y_test, pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")