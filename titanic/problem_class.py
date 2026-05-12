import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Искусственно создадим данные для демонстрации проблемы категорий в данных
data = pd.DataFrame ({
    "city": ["Moscow", "Spb", "Kazan", "Moscow"],
    "price" : [100, 150, 120, 110],
    "rooms" : [2, 3, 2, 3],
})

y = [0, 1, 0, 1]
X = data[["city", "price", "rooms"]]
model = DecisionTreeClassifier()
model.fit(X, y)
# На строчке выдаст ошибку, так как "Moscow" не соответсвует значению float